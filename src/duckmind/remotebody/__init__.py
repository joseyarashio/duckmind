"""The remote body: the brain on the home computer, the body (robotd) on the duck, the LAN between.

* :mod:`~duckmind.remotebody.link` — the wire: token handshake, signed frames, heartbeat constants;
* :mod:`~duckmind.remotebody.bridge` — the body side: robotd (tofd, mediad) for one brain, and the
  deadman that stops and sits the duck when the brain's link goes quiet;
* :mod:`~duckmind.remotebody.client` — the brain side: the body's daemons as local sockets;
* :mod:`~duckmind.remotebody.simbody` — a simulated body behind the bridge (the rehearsal);
* :mod:`~duckmind.remotebody.netshim` — a delay / jitter / loss / cut shim for testing on one machine.

``python -m duckmind.remotebody token|bridge|sim-body|shim --help``; the steps for a real duck are
in ``docs/real-microduck.md``.
"""
