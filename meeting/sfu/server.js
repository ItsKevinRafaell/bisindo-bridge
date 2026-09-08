#!/usr/bin/env node

'use strict';

const http = require('http');
const { Server: SocketIOServer } = require('socket.io');
const mediasoup = require('mediasoup');

const PORT = Number(process.env.SFU_PORT) || 4501;

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

/** @type {mediasoup.types.Worker} */
let worker;

/** roomId → mediasoup.Router */
const rooms = new Map();

/**
 * peerId → {
 *   socket: Socket,
 *   roomId: string,
 *   sendTransport: mediasoup.types.WebRtcTransport | null,
 *   recvTransport: mediasoup.types.WebRtcTransport | null,
 *   producers: Map<producerId, mediasoup.types.Producer>,
 *   consumers: Map<consumerId, mediasoup.types.Consumer>,
 * }
 */
const peers = new Map();

// ---------------------------------------------------------------------------
// mediasoup Router media codecs
// ---------------------------------------------------------------------------

const mediaCodecs = [
  {
    kind: 'audio',
    mimeType: 'audio/opus',
    clockRate: 48000,
    channels: 2,
  },
  {
    kind: 'video',
    mimeType: 'video/VP8',
    clockRate: 90000,
    parameters: {
      'x-google-start-bitrate': 1000,
    },
  },
];

// ---------------------------------------------------------------------------
// WebRTC transport listen IPs
// ---------------------------------------------------------------------------

const webRtcTransportOptions = {
  listenIps: [
    {
      ip: process.env.MEDIASOUP_LISTEN_IP || '0.0.0.0',
      announcedIp: process.env.MEDIASOUP_ANNOUNCED_IP || null,
    },
  ],
  initialAvailableOutgoingBitrate: 1_000_000,
  minimumAvailableOutgoingBitrate: 200_000,
  maxSctpMessageSize: 262144,
  enableUdp: true,
  enableTcp: true,
  preferUdp: true,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getOrCreateRoom(roomId) {
  if (rooms.has(roomId)) {
    return Promise.resolve(rooms.get(roomId));
  }
  return worker.createRouter({ mediaCodecs }).then((router) => {
    rooms.set(roomId, router);

    router.on('workerclose', () => {
      rooms.delete(roomId);
    });

    return router;
  });
}

function getPeer(socket) {
  return peers.get(socket.id);
}

function broadcastRoom(roomId, event, data, excludeSocketId) {
  for (const [peerId, peer] of peers) {
    if (peer.roomId === roomId && peerId !== excludeSocketId) {
      peer.socket.emit(event, data);
    }
  }
}

function peerListForRoom(roomId) {
  const list = [];
  for (const [peerId, peer] of peers) {
    if (peer.roomId === roomId) {
      list.push({
        peerId,
        producers: [...peer.producers.keys()],
      });
    }
  }
  return list;
}

async function closePeer(peerId) {
  const peer = peers.get(peerId);
  if (!peer) return;

  // Close all consumers
  for (const consumer of peer.consumers.values()) {
    consumer.close();
  }
  peer.consumers.clear();

  // Close all producers
  for (const producer of peer.producers.values()) {
    producer.close();
  }
  peer.producers.clear();

  // Close transports
  if (peer.sendTransport) {
    peer.sendTransport.close();
  }
  if (peer.recvTransport) {
    peer.recvTransport.close();
  }

  // Notify others in the room
  broadcastRoom(peer.roomId, 'peerLeft', { peerId });

  peers.delete(peerId);
}

function cleanupEmptyRoom(roomId) {
  for (const peer of peers.values()) {
    if (peer.roomId === roomId) return; // still has peers
  }
  const router = rooms.get(roomId);
  if (router) {
    router.close();
    rooms.delete(roomId);
  }
}

// ---------------------------------------------------------------------------
// Socket.IO event handlers
// ---------------------------------------------------------------------------

function registerHandlers(socket) {
  // ---- getRouterRtpCapabilities ----
  socket.on('getRouterRtpCapabilities', async ({ roomId }, callback) => {
    try {
      const router = await getOrCreateRoom(roomId);
      callback({ rtpCapabilities: router.rtpCapabilities });
    } catch (err) {
      callback({ error: err.message });
    }
  });

  // ---- join ----
  socket.on('join', async ({ roomId, peerId }, callback) => {
    try {
      const router = await getOrCreateRoom(roomId);

      // Store peer (peerId defaults to socket.id if not supplied)
      const id = peerId || socket.id;
      if (peers.has(id)) {
        // Reconnect scenario — close old peer first
        await closePeer(id);
      }

      peers.set(id, {
        socket,
        roomId,
        sendTransport: null,
        recvTransport: null,
        producers: new Map(),
        consumers: new Map(),
      });

      // Remap socket.id → peer id lookup
      socket.peerId = id;
      socket.roomId = roomId;

      const existingPeers = peerListForRoom(roomId).filter((p) => p.peerId !== id);

      callback({
        rtpCapabilities: router.rtpCapabilities,
        peers: existingPeers,
      });

      // Notify others
      broadcastRoom(roomId, 'peerJoined', { peerId: id }, socket.id);
    } catch (err) {
      callback({ error: err.message });
    }
  });

  // ---- createWebRtcTransport ----
  socket.on('createWebRtcTransport', async ({ roomId, forceTcp = false, producing, consuming }, callback) => {
    try {
      const router = rooms.get(roomId);
      if (!router) return callback({ error: 'room not found' });

      const peer = peers.get(socket.peerId || socket.id);
      if (!peer) return callback({ error: 'peer not found — call join first' });

      const transportOpts = {
        ...webRtcTransportOptions,
        enableTcp: forceTcp || webRtcTransportOptions.enableTcp,
        enableUdp: !forceTcp && webRtcTransportOptions.enableUdp,
      };

      const transport = await router.createWebRtcTransport(transportOpts);

      if (producing) {
        peer.sendTransport = transport;
      } else if (consuming) {
        peer.recvTransport = transport;
      }

      transport.on('dtlsstatechange', (dtlsState) => {
        if (dtlsState === 'closed' || dtlsState === 'failed') {
          transport.close();
        }
      });

      callback({
        transportOptions: {
          id: transport.id,
          iceParameters: transport.iceParameters,
          iceCandidates: transport.iceCandidates,
          dtlsParameters: transport.dtlsParameters,
          sctpParameters: transport.sctpParameters,
        },
      });
    } catch (err) {
      callback({ error: err.message });
    }
  });

  // ---- connectTransport ----
  socket.on('connectTransport', async ({ transportId, dtlsParameters }, callback) => {
    try {
      const peer = peers.get(socket.peerId || socket.id);
      if (!peer) return callback({ error: 'peer not found' });

      const transport =
        (peer.sendTransport && peer.sendTransport.id === transportId && peer.sendTransport) ||
        (peer.recvTransport && peer.recvTransport.id === transportId && peer.recvTransport) ||
        null;

      if (!transport) return callback({ error: 'transport not found' });

      await transport.connect({ dtlsParameters });
      callback({ connected: true });
    } catch (err) {
      callback({ error: err.message });
    }
  });

  // ---- produce ----
  socket.on('produce', async ({ transportId, kind, rtpParameters, appData }, callback) => {
    try {
      const peer = peers.get(socket.peerId || socket.id);
      if (!peer) return callback({ error: 'peer not found' });

      if (!peer.sendTransport || peer.sendTransport.id !== transportId) {
        return callback({ error: 'invalid send transport' });
      }

      const producer = await peer.sendTransport.produce({
        kind,
        rtpParameters,
        appData: { ...appData, peerId: socket.peerId || socket.id },
      });

      peer.producers.set(producer.id, producer);

      producer.on('transportclose', () => {
        peer.producers.delete(producer.id);
        broadcastRoom(peer.roomId, 'producerClosed', {
          producerId: producer.id,
          peerId: socket.peerId || socket.id,
        });
      });

      // Notify others about new producer so they can consume
      broadcastRoom(peer.roomId, 'newProducer', {
        peerId: socket.peerId || socket.id,
        producerId: producer.id,
        kind: producer.kind,
        appData: producer.appData,
      }, socket.id);

      callback({ id: producer.id });
    } catch (err) {
      callback({ error: err.message });
    }
  });

  // ---- consume ----
  socket.on('consume', async ({ peerId: remotePeerId, producerId, rtpCapabilities }, callback) => {
    try {
      const peer = peers.get(socket.peerId || socket.id);
      if (!peer) return callback({ error: 'peer not found' });

      const router = rooms.get(peer.roomId);
      if (!router) return callback({ error: 'room not found' });

      if (!router.canConsume({ producerId, rtpCapabilities })) {
        return callback({ error: 'cannot consume this producer' });
      }

      if (!peer.recvTransport) {
        return callback({ error: 'no recv transport — call createWebRtcTransport first' });
      }

      const consumer = await peer.recvTransport.consume({
        producerId,
        rtpCapabilities,
        paused: false,
      });

      peer.consumers.set(consumer.id, consumer);

      consumer.on('transportclose', () => {
        peer.consumers.delete(consumer.id);
      });

      consumer.on('producerclose', () => {
        peer.consumers.delete(consumer.id);
        socket.emit('producerClosed', { producerId, peerId: remotePeerId });
      });

      callback({
        consumerId: consumer.id,
        producerId,
        kind: consumer.kind,
        rtpParameters: consumer.rtpParameters,
      });
    } catch (err) {
      callback({ error: err.message });
    }
  });

  // ---- closeProducer ----
  socket.on('closeProducer', async ({ producerId }, callback) => {
    try {
      const peer = peers.get(socket.peerId || socket.id);
      if (!peer) return callback({ error: 'peer not found' });

      const producer = peer.producers.get(producerId);
      if (!producer) return callback({ error: 'producer not found' });

      producer.close();
      peer.producers.delete(producerId);

      broadcastRoom(peer.roomId, 'producerClosed', {
        producerId,
        peerId: socket.peerId || socket.id,
      });

      callback({ closed: true });
    } catch (err) {
      callback({ error: err.message });
    }
  });

  // ---- closeConsumer ----
  socket.on('closeConsumer', async ({ consumerId }, callback) => {
    try {
      const peer = peers.get(socket.peerId || socket.id);
      if (!peer) return callback({ error: 'peer not found' });

      const consumer = peer.consumers.get(consumerId);
      if (!consumer) return callback({ error: 'consumer not found' });

      consumer.close();
      peer.consumers.delete(consumerId);

      callback({ closed: true });
    } catch (err) {
      callback({ error: err.message });
    }
  });

  // ---- pauseProducer / resumeProducer ----
  socket.on('pauseProducer', async ({ producerId }, callback) => {
    try {
      const peer = peers.get(socket.peerId || socket.id);
      if (!peer) return callback({ error: 'peer not found' });
      const producer = peer.producers.get(producerId);
      if (!producer) return callback({ error: 'producer not found' });
      await producer.pause();
      broadcastRoom(peer.roomId, 'producerPaused', { producerId, peerId: socket.peerId || socket.id });
      callback({ paused: true });
    } catch (err) {
      callback({ error: err.message });
    }
  });

  socket.on('resumeProducer', async ({ producerId }, callback) => {
    try {
      const peer = peers.get(socket.peerId || socket.id);
      if (!peer) return callback({ error: 'peer not found' });
      const producer = peer.producers.get(producerId);
      if (!producer) return callback({ error: 'producer not found' });
      await producer.resume();
      broadcastRoom(peer.roomId, 'producerResumed', { producerId, peerId: socket.peerId || socket.id });
      callback({ resumed: true });
    } catch (err) {
      callback({ error: err.message });
    }
  });

  // ---- disconnect ----
  socket.on('disconnect', async () => {
    const peerId = socket.peerId || socket.id;
    const peer = peers.get(peerId);
    const roomId = peer ? peer.roomId : null;
    await closePeer(peerId);
    if (roomId) cleanupEmptyRoom(roomId);
  });
}

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

async function start() {
  // 1. Create mediasoup worker
  worker = await mediasoup.createWorker({
    logLevel: process.env.MEDIASOUP_LOG_LEVEL || 'warn',
    logTags: ['info', 'ice', 'dtls', 'rtp', 'srtp', 'rtcp'],
    rtcMinPort: Number(process.env.MEDIASOUP_MIN_PORT) || 40000,
    rtcMaxPort: Number(process.env.MEDIASOUP_MAX_PORT) || 49999,
  });

  worker.on('died', () => {
    console.error('[sfu] mediasoup Worker died — exiting in 2s');
    setTimeout(() => process.exit(1), 2000);
  });

  console.log(`[sfu] mediasoup Worker created (pid ${worker.pid})`);

  // 2. HTTP + Socket.IO
  const httpServer = http.createServer();
  const io = new SocketIOServer(httpServer, {
    cors: {
      origin: '*',
      methods: ['GET', 'POST'],
    },
    maxHttpBufferSize: 1e6,
  });

  io.on('connection', (socket) => {
    console.log(`[sfu] socket connected: ${socket.id}`);
    registerHandlers(socket);
  });

  httpServer.listen(PORT, () => {
    console.log(`[sfu] Socket.IO signaling server listening on port ${PORT}`);
  });
}

start().catch((err) => {
  console.error('[sfu] failed to start:', err);
  process.exit(1);
});
