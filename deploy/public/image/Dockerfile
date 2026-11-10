# syntax=docker/dockerfile:1.7
#
# duckmind public play: one simulated duck for one visitor (deploy/public/image/; NON-COMMERCIAL ONLY:
# the Microduck 3D meshes are CC BY-NC-SA 4.0 and are fetched here at build time, never committed).
#
#   docker build -f deploy/public/image/Dockerfile .          (context: the repository root, or `build.sh`)
#
# Stages: fetch (upstream checkouts and Hugging Face networks, every one pinned and verified),
# robotd (Pollen's robotd 0.18.5 + duckmind's poseJoints patch, and tofd), wheels (duckmind and its
# Python packages; pupil-apriltags has no arm64 wheel and is compiled there),
# app (Python 3.12, MuJoCo 3.10.0, ONNX Runtime 1.27.0; non-root; app files read-only; /tmp writable).
# Builds for linux/amd64 (Cloudflare Containers) and linux/arm64.

ARG PYTHON_IMAGE=python:3.12-slim-bookworm
ARG PYTHON_BUILD_IMAGE=python:3.12-bookworm
ARG RUST_IMAGE=rust:1.89-bookworm

# ── fetch ─────────────────────────────────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS fetch
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Every pin (upstream commits and tag, the Hugging Face repos and revisions, each file's sha256) is in
# src/duckmind/pins/, the files `python -m duckmind.setup_cmd` reads too: Pollen's microduck at the
# daemon-v0.18.5 tag (robotd; Apache-2.0), microduck_rl at duckmind's pinned commit (the MJCF and body
# server, Apache-2.0; the 3D meshes, CC BY-NC-SA 4.0), Pollen's policy set v9 and duckmind's seven
# trained motions (Hugging Face, Apache-2.0).
WORKDIR /fetch
COPY src/duckmind/pins/upstream.env src/duckmind/pins/hub.py src/duckmind/pins/policies-v9.sha256 \
     src/duckmind/pins/motions.sha256 ./
RUN . ./upstream.env \
    && git init -q microduck \
    && git -C microduck fetch -q --depth 1 "${MICRODUCK_URL}" "refs/tags/${MICRODUCK_TAG}" \
    && git -C microduck -c advice.detachedHead=false checkout -q FETCH_HEAD \
    && test "$(git -C microduck rev-parse HEAD)" = "${MICRODUCK_COMMIT}" \
    && rm -rf microduck/.git
RUN . ./upstream.env \
    && git init -q rl \
    && git -C rl fetch -q --depth 1 "${MICRODUCK_RL_URL}" "${MICRODUCK_RL_COMMIT}" \
    && git -C rl -c advice.detachedHead=false checkout -q FETCH_HEAD \
    && test "$(git -C rl rev-parse HEAD)" = "${MICRODUCK_RL_COMMIT}" \
    && mkdir -p microduck_rl/src/mjlab_microduck \
    && cp -R rl/src/mjlab_microduck/__init__.py rl/src/mjlab_microduck/robot rl/src/mjlab_microduck/sim \
         microduck_rl/src/mjlab_microduck/ \
    && cp rl/LICENSE rl/README.md microduck_rl/ \
    && if [ -f rl/LICENSE-MODELS.md ]; then cp rl/LICENSE-MODELS.md microduck_rl/; fi \
    && rm -rf rl
RUN . ./upstream.env && printf '%s\n' \
    "The 3D meshes (.stl) under src/mjlab_microduck/robot/microduck/assets/ are the Microduck 3D model by" \
    "Pollen Robotics (https://github.com/pollen-robotics/microduck_rl, commit ${MICRODUCK_RL_COMMIT})," \
    "licensed under CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/). They were fetched" \
    "unmodified when this image was built and are used here for non-commercial use only. The code and the" \
    "MJCF robot descriptions are Apache-2.0 (LICENSE)." > microduck_rl/MESHES-LICENCE.txt
RUN . ./upstream.env \
    && python hub.py "${POLICIES_REPO}" "${POLICIES_REV}" policies-v9.sha256 policies/v9 \
    && python hub.py "${MOTIONS_REPO}" "${MOTIONS_REV}" motions.sha256 motions-fetched \
    && printf 'repo=%s\nversion=%s\n' "${POLICIES_REPO}" "${POLICIES_REV}" > policies/v9/.source

# ── robotd ────────────────────────────────────────────────────────────────────────────────────────
FROM ${RUST_IMAGE} AS robotd
# dev = the 0.18.5 recipe's debug build (what the simulator was measured on); release is an option
ARG ROBOTD_PROFILE=dev
COPY --from=fetch /fetch/microduck /build/microduck
COPY src/duckmind/pins/upstream.env /build/
COPY third_party/microduck_emotions/robotd-poseJoints-0.18.5.patch /build/third_party/microduck_emotions/
WORKDIR /build/microduck
RUN . ../upstream.env && git apply --check "../${MICRODUCK_PATCH}" && git apply "../${MICRODUCK_PATCH}"
# robotd and tofd, and the sounds tool the page's voice bank is rendered with (duckmind.simweb.voice looks
# for it in the checkout's target/debug); debug symbols stripped (the code is the recipe's build)
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/build/microduck/target,sharing=locked \
    cargo build --locked --profile "${ROBOTD_PROFILE}" -p robotd --bin robotd \
    && cargo build --locked --profile "${ROBOTD_PROFILE}" -p tof --bin tofd \
    && cargo build --locked --profile "${ROBOTD_PROFILE}" -p sounds --bin sounds \
    && dir="$( [ "${ROBOTD_PROFILE}" = dev ] && echo debug || echo "${ROBOTD_PROFILE}" )" \
    && mkdir -p /out/bin /out/sounds \
    && strip --strip-debug -o /out/bin/robotd "target/${dir}/robotd" \
    && strip --strip-debug -o /out/bin/tofd "target/${dir}/tofd" \
    && strip --strip-debug -o /out/sounds/sounds "target/${dir}/sounds"

# ── wheels ────────────────────────────────────────────────────────────────────────────────────────
FROM ${PYTHON_BUILD_IMAGE} AS wheels
RUN apt-get update && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/public/image/requirements.txt /src/requirements.txt
RUN pip wheel --no-cache-dir -w /wheels -r /src/requirements.txt
WORKDIR /src
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN pip wheel --no-deps --no-cache-dir -w /wheels .

# ── app ───────────────────────────────────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS app
# tini: PID 1 forwards SIGTERM and reaps robotd; Mesa: MuJoCo's offscreen camera renders on the CPU (EGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
        tini libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libgles2 libgbm1 \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/public/image/requirements.txt /tmp/requirements.txt
RUN --mount=type=bind,from=wheels,source=/wheels,target=/wheels \
    pip install --no-cache-dir --no-index --find-links /wheels -r /tmp/requirements.txt \
    && pip install --no-cache-dir --no-index --no-deps /wheels/duckmind-*.whl \
    && rm /tmp/requirements.txt
COPY --from=fetch /fetch/microduck /opt/duckmind/upstream/microduck
COPY --from=fetch /fetch/microduck_rl /opt/duckmind/upstream/microduck_rl
COPY --from=fetch /fetch/policies /opt/duckmind/policies
COPY --from=robotd /out/bin/ /opt/duckmind/bin/
COPY --from=robotd /out/sounds/sounds /opt/duckmind/upstream/microduck/target/debug/sounds
COPY deploy/public/image/entrypoint.sh /opt/duckmind/
# duckmind's motion table on v9 (duckmind.pins.motions, the one setup writes; guard caps: pins/motion-guard.json)
RUN --mount=type=bind,from=fetch,source=/fetch/motions-fetched,target=/motions-fetched \
    python -m duckmind.pins.motions /motions-fetched /opt/duckmind/policies/v9 /opt/duckmind/motions \
    && python -m compileall -q /opt/duckmind/upstream/microduck_rl/src \
    && chown -R root:root /opt/duckmind && chmod -R a+rX,go-w /opt/duckmind \
    && chmod 0755 /opt/duckmind/entrypoint.sh /opt/duckmind/bin/robotd /opt/duckmind/bin/tofd \
    && useradd --uid 10001 --user-group --no-create-home --home-dir /tmp/home --shell /usr/sbin/nologin duck
# One thread each for Mesa's software renderer and BLAS: under a CPU quota a renderer that fans out over
# every core spends the quota in bursts and stalls the physics (measured 2026-11-10 at 1 vCPU: median
# realtime 0.84 -> 0.97, throttled time 528 s -> 201 s in 10 min)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp/home \
    DUCKMIND_HOME=/tmp/duckmind \
    DUCKMIND_UPSTREAM=/opt/duckmind/upstream \
    DUCKMIND_POLICY_DIR=/opt/duckmind/policies/v9 \
    DUCKMIND_ROBOTD_SKILLS=/opt/duckmind/motions/skills.json \
    DUCKMIND_LASTMAP=soft \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    LP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    DUCKMIND_PUBLIC_MINUTES=15
USER duck
WORKDIR /tmp
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=240s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/ping', timeout=2)"]
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/duckmind/entrypoint.sh"]
