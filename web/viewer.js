import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";

const KM_SCALE = 1 / 1000;         // km -> scene units
const EARTH_RADIUS_KM = 6378.137;
const OMEGA_EARTH_RAD_S = 7.2921150e-5;

let scene, camera, renderer, controls;
let earthMesh, earthGroup;
let cloudSphere = null;
let nightSphere = null;
let satMeshes = new Map();
let satTrails = new Map();
let gsMeshes = new Map();
let coverageMesh = null;
let coverageData = null;

let samplesBySat = {};
let gsData = {};
let visData = {};                  // key "GS|SAT" -> [0/1...]
let times = [];

let isPlaying = false;
let frameIndex = 0;
let playbackFrameCounter = 0;  // controls playback speed

let selectedSatId = null;
let selectedGsId = null;

let followEnabled = false;
let followDistance = 10; // slider units (scaled later)
let userZoomDistance = null; // track user's manual zoom

let coverageEnabled = true;

let heatmapSphere = null;
let heatmapTexture = null;
let heatmapEnabled = false;

let sunLight = null;
const sunDir = new THREE.Vector3(1, 0, 0).normalize(); // you can change this live later

let lastHover = { i: null, j: null, lat: null, lon: null };

// HUD
const playBtn = document.getElementById("playBtn");
const pauseBtn = document.getElementById("pauseBtn");
const timeSlider = document.getElementById("timeSlider");
const timeLabel = document.getElementById("timeLabel");
const clearSelectionBtn = document.getElementById("clearSelectionBtn");
const selectedSatEl = document.getElementById("selectedSat");
const selectedGsEl = document.getElementById("selectedGs");
const visModeLabel = document.getElementById("visModeLabel");
const visList = document.getElementById("visList");
const followBtn = document.getElementById("followBtn");
const hoverInfo = document.getElementById("hoverInfo");
const coverageSelect = document.getElementById("coverageSelect");
const toggleCoverageBtn = document.getElementById("toggleCoverageBtn");
const satSelect = document.getElementById("satSelect");

playBtn.onclick = () => isPlaying = true;
pauseBtn.onclick = () => isPlaying = false;
clearSelectionBtn.onclick = () => {
  selectedSatId = null;
  selectedGsId = null;
  followEnabled = false;
  followBtn.textContent = "Follow: OFF";
  satSelect.value = "";  // clear dropdown
  updateHud();
};

timeSlider.oninput = () => {
  isPlaying = false;
  frameIndex = parseInt(timeSlider.value, 10);
  updateFrame(frameIndex);
};

followBtn.onclick = () => {
  // Only meaningful if a sat is selected
  if (!selectedSatId) return;
  followEnabled = !followEnabled;
  followBtn.textContent = `Follow: ${followEnabled ? "ON" : "OFF"}`;
};

toggleCoverageBtn.onclick = () => {
  heatmapEnabled = !heatmapEnabled;
  toggleCoverageBtn.textContent = `Coverage: ${heatmapEnabled ? "ON" : "OFF"}`;
  rebuildHeatmapSurface();
};

coverageSelect.onchange = async () => {
  await loadCoverage(coverageSelect.value);
};

satSelect.onchange = () => {
  const sid = satSelect.value;
  if (sid) {
    selectedSatId = sid;
    followEnabled = true;
    followBtn.textContent = "Follow: ON";
  } else {
    selectedSatId = null;
    followEnabled = false;
    followBtn.textContent = "Follow: OFF";
  }
  updateFrame(frameIndex);
};

function clamp01(x){ return Math.max(0, Math.min(1, x)); }

function heatColorRGB(t) {
  // blue -> cyan -> green -> yellow -> red
  t = clamp01(t);
  let r=0,g=0,b=0;
  if (t < 0.25) { const k=t/0.25; r=0; g=k; b=1; }
  else if (t < 0.5) { const k=(t-0.25)/0.25; r=0; g=1; b=1-k; }
  else if (t < 0.5) { const k=(t-0.5)/0.25; r=k; g=1; b=0; }
  else { const k=(t-0.75)/0.25; r=1; g=1-k; b=0; }
  return {r,g,b};
}

// Equirectangular: width = 2*height usually
function buildHeatmapCanvasTexture(coverageData, width=2048, height=1024) {
  const latBins = coverageData.lat_bins;
  const lonBins = coverageData.lon_bins;
  const hits = coverageData.hits;

  // Find max hits
  let maxHits = 1;
  for (let i=0;i<hits.length;i++){
    for (let j=0;j<hits[i].length;j++){
      if (hits[i][j] > maxHits) maxHits = hits[i][j];
    }
  }

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(width, height);
  const data = img.data;

  // We'll paint each grid cell as a rectangle in pixel space.
  // Map lon [-180..180] -> x [0..W)
  // Map lat [-90..90]  -> y [H-1..0] (flip so north is up)
  const latStep = coverageData.lat_step_deg;
  const lonStep = coverageData.lon_step_deg;

  function lonToX(lonDeg) {
    return Math.floor(((lonDeg + 180) / 360) * width);
  }
  function latToY(latDeg) {
    return Math.floor((1 - (latDeg + 90) / 180) * height);
  }

  // Paint background transparent
  for (let k=0;k<data.length;k+=4) data[k+3] = 0;

  for (let i=0;i<latBins.length;i++){
    for (let j=0;j<lonBins.length;j++){
      const h = hits[i][j];
      if (h <= 0) continue;

      // normalize & boost contrast
      let t = h / maxHits;
      t = Math.pow(t, 0.6);

      const {r,g,b} = heatColorRGB(t);

      // alpha: stronger where hotter (t)
      const a = Math.floor(40 + 215 * t); // 40..255

      // Cell bounds (center +/- step/2)
      const latC = latBins[i];
      const lonC = lonBins[j];

      const lat0 = latC - latStep/2;
      const lat1 = latC + latStep/2;
      const lon0 = lonC - lonStep/2;
      const lon1 = lonC + lonStep/2;

      // Convert to pixel bounds
      let x0 = lonToX(lon0);
      let x1 = lonToX(lon1);
      let y0 = latToY(lat1);
      let y1 = latToY(lat0);

      // Handle wrap across dateline
      const drawRect = (xa, xb) => {
        xa = Math.max(0, Math.min(width-1, xa));
        xb = Math.max(0, Math.min(width, xb));
        y0 = Math.max(0, Math.min(height-1, y0));
        y1 = Math.max(0, Math.min(height, y1));

        for (let y=y0; y<y1; y++){
          for (let x=xa; x<xb; x++){
            const idx = (y*width + x)*4;
            data[idx+0] = Math.floor(r*255);
            data[idx+1] = Math.floor(g*255);
            data[idx+2] = Math.floor(b*255);
            data[idx+3] = a;
          }
        }
      };

      if (x1 >= x0) {
        drawRect(x0, x1);
      } else {
        // wrapped
        drawRect(0, x1);
        drawRect(x0, width);
      }
    }
  }

  ctx.putImageData(img, 0, 0);

  // OPTIONAL: soften pixel edges slightly
  // ctx.filter = "blur(1px)";
  // const tmp = ctx.getImageData(0,0,width,height);
  // ctx.putImageData(tmp,0,0);

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;

  return tex;
}

function ensureHeatmapSphere() {
  if (heatmapSphere) return;

  const r = EARTH_RADIUS_KM * KM_SCALE * 1.004; // slightly above Earth surface
  const geom = new THREE.SphereGeometry(r, 64, 64);

  const mat = new THREE.MeshBasicMaterial({
    map: null,
    transparent: true,
    opacity: 1.0,
    depthWrite: false,
    blending: THREE.AdditiveBlending
  });

  heatmapSphere = new THREE.Mesh(geom, mat);
  heatmapSphere.userData = { type: "heatmap_sphere" };

  // attach to earthGroup so it rotates with Earth
  earthGroup.add(heatmapSphere);
}

function ensureCloudSphere() {
  if (cloudSphere) return;

  const r = EARTH_RADIUS_KM * KM_SCALE * 1.01; // slightly above Earth
  const geom = new THREE.SphereGeometry(r, 64, 64);

  const mat = new THREE.MeshStandardMaterial({
    transparent: true,
    opacity: 0.75,
    depthWrite: false
  });

  cloudSphere = new THREE.Mesh(geom, mat);

  const loader = new THREE.TextureLoader();
  loader.load("./assets/earth_clouds.png", (tex) => {
    tex.colorSpace = THREE.SRGBColorSpace;
    cloudSphere.material.map = tex;
    cloudSphere.material.needsUpdate = true;
  });

  earthGroup.add(cloudSphere);
}

function addAtmosphereGlow() {
  const r = EARTH_RADIUS_KM * KM_SCALE * 1.025;
  const geom = new THREE.SphereGeometry(r, 64, 64);

  const mat = new THREE.MeshBasicMaterial({
    color: 0x66aaff,
    transparent: true,
    opacity: 0.35,
    blending: THREE.AdditiveBlending,
    side: THREE.BackSide
  });

  const glow = new THREE.Mesh(geom, mat);
  earthGroup.add(glow);
}

function ensureNightLightsSphere() {
  if (nightSphere) return;

  const r = EARTH_RADIUS_KM * KM_SCALE * 1.006; // just above day texture, below clouds ideally
  const geom = new THREE.SphereGeometry(r, 64, 64);

  const loader = new THREE.TextureLoader();
  const nightTex = loader.load("./assets/earth_night.png", (t) => {
    t.colorSpace = THREE.SRGBColorSpace;
  });

  // Shader: show night texture only on the dark side (N·S < 0)
  const mat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uNightTex: { value: nightTex },
      uSunDir: { value: sunDir.clone() },   // will be updated each frame
      uIntensity: { value: 1.2 },
      uSoftness: { value: 0.18 }            // terminator softness
    },
    vertexShader: `
      varying vec3 vNormalW;
      varying vec2 vUv;
      void main() {
        vUv = uv;
        vNormalW = normalize(mat3(modelMatrix) * normal);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform sampler2D uNightTex;
      uniform vec3 uSunDir;
      uniform float uIntensity;
      uniform float uSoftness;
      varying vec3 vNormalW;
      varying vec2 vUv;

      void main() {
        vec3 N = normalize(vNormalW);
        vec3 S = normalize(uSunDir);

        // dot > 0 = day, dot < 0 = night
        float nds = dot(N, S);

        // Night mask: 1 on deep night, 0 on day, softened near terminator
        // nds in [-1..1]. We want mask high when nds is negative.
        float night = smoothstep(uSoftness, -uSoftness, nds);

        vec4 tex = texture2D(uNightTex, vUv);

        // Use texture brightness; keep alpha controlled by night mask
        vec3 col = tex.rgb * uIntensity;
        float a = night * 0.9;

        gl_FragColor = vec4(col, a);
      }
    `
  });

  nightSphere = new THREE.Mesh(geom, mat);
  earthGroup.add(nightSphere);
}

function rebuildHeatmapSurface() {
  ensureHeatmapSphere();

  if (!heatmapEnabled || !coverageData) {
    heatmapSphere.visible = false;
    return;
  }

  heatmapSphere.visible = true;

  // dispose old texture
  if (heatmapTexture) {
    heatmapTexture.dispose();
    heatmapTexture = null;
  }

  heatmapTexture = buildHeatmapCanvasTexture(coverageData, 2048, 1024);
  heatmapSphere.material.map = heatmapTexture;
  heatmapSphere.material.needsUpdate = true;
}

async function loadBundle() {
  const res = await fetch("../out/playback_bundle.json");
  const data = await res.json();

  samplesBySat = data.sat_positions_eci_km || {};
  gsData = data.ground_stations || {};
  visData = data.visibility || {};
  times = data.times_s || [];

  const satIds = Object.keys(samplesBySat);
  if (satIds.length === 0) throw new Error("No satellites in bundle.");
  if (times.length === 0) throw new Error("No times in bundle.");

  // Populate satellite selector dropdown
  satIds.forEach(sid => {
    const option = document.createElement("option");
    option.value = sid;
    option.textContent = sid;
    satSelect.appendChild(option);
  });

  timeSlider.max = String(times.length - 1);
  timeSlider.value = "0";
}

async function loadCoverage(path) {
  const res = await fetch(path);
  coverageData = await res.json();
  rebuildHeatmapSurface();   // <- new
}

function initThree() {
  scene = new THREE.Scene();

  camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.001, 1e9);
  camera.position.set(0, 0, EARTH_RADIUS_KM * KM_SCALE * 8);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.screenSpacePanning = false;
  controls.minDistance = EARTH_RADIUS_KM * KM_SCALE * 1.2;
  controls.maxDistance = EARTH_RADIUS_KM * KM_SCALE * 50;
  controls.rotateSpeed = 0.5;
  controls.zoomSpeed = 1.2;
  controls.panSpeed = 0.8;

  // Lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  sunLight = new THREE.DirectionalLight(0xffffff, 1.1);
  sunLight.position.set(sunDir.x, sunDir.y, sunDir.z);
  scene.add(sunLight);

  // Group that rotates with Earth
  earthGroup = new THREE.Group();
  scene.add(earthGroup);

  // Earth
  const earthGeom = new THREE.SphereGeometry(EARTH_RADIUS_KM * KM_SCALE, 48, 48);
  const earthMat = new THREE.MeshStandardMaterial({
    roughness: 1.0,
    metalness: 0.0
  });

  earthMesh = new THREE.Mesh(earthGeom, earthMat);
  earthGroup.add(earthMesh);

  // Load day texture (served from /web/assets/)
  const loader = new THREE.TextureLoader();
  loader.load("./assets/earth_day.jpg", (tex) => {
    tex.colorSpace = THREE.SRGBColorSpace;   // important for correct colors (newer three)
    earthMesh.material.map = tex;
    earthMesh.material.needsUpdate = true;
  });

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
}

function latLonToECEF(latDeg, lonDeg, altKm) {
  const lat = latDeg * Math.PI / 180;
  const lon = lonDeg * Math.PI / 180;
  const r = (EARTH_RADIUS_KM + (altKm || 0)) * KM_SCALE;

  const clat = Math.cos(lat), slat = Math.sin(lat);
  const clon = Math.cos(lon), slon = Math.sin(lon);

  const x = r * clat * clon;
  const y = r * clat * slon;
  const z = r * slat;
  return new THREE.Vector3(x, y, z);
}

function vecToLatLonDeg(v) {
  // v is a THREE.Vector3 in EarthGroup coordinates (ECEF-like)
  const r = v.length();
  if (r === 0) return { lat: 0, lon: 0 };

  const lat = Math.asin(v.z / r) * 180 / Math.PI;
  const lon = Math.atan2(v.y, v.x) * 180 / Math.PI;
  // wrap lon to [-180,180)
  const lonWrap = ((lon + 180) % 360) - 180;
  return { lat, lon: lonWrap };
}

function nearestIndex(bins, value) {
  // bins is sorted centers; return closest index
  let bestI = 0;
  let bestD = Infinity;
  for (let i = 0; i < bins.length; i++) {
    const d = Math.abs(bins[i] - value);
    if (d < bestD) {
      bestD = d;
      bestI = i;
    }
  }
  return bestI;
}

function heatColor(t) {
  // t in [0,1] -> RGB
  // blue -> cyan -> green -> yellow -> red
  t = Math.max(0, Math.min(1, t));

  let r = 0, g = 0, b = 0;

  if (t < 0.25) {
    // blue -> cyan
    const k = t / 0.25;
    r = 0; g = k; b = 1;
  } else if (t < 0.5) {
    // cyan -> green
    const k = (t - 0.25) / 0.25;
    r = 0; g = 1; b = 1 - k;
  } else if (t < 0.75) {
    // green -> yellow
    const k = (t - 0.5) / 0.25;
    r = k; g = 1; b = 0;
  } else {
    // yellow -> red
    const k = (t - 0.75) / 0.25;
    r = 1; g = 1 - k; b = 0;
  }

  return { r, g, b };
}

function createSatObjects() {
  const satIds = Object.keys(samplesBySat);

  for (const sid of satIds) {
    // Marker
    const geom = new THREE.SphereGeometry(80 * KM_SCALE, 16, 16);
    const mat = new THREE.MeshStandardMaterial({ color: 0xffaa00 });
    const mesh = new THREE.Mesh(geom, mat);
    mesh.userData = { type: "sat", sid };
    scene.add(mesh);
    satMeshes.set(sid, mesh);

    // Trail
    const lineGeom = new THREE.BufferGeometry();
    const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff });
    const line = new THREE.Line(lineGeom, lineMat);
    line.userData = { type: "sat_trail", sid };
    scene.add(line);
    satTrails.set(sid, line);
  }
}

function createGroundStations() {
  for (const gsId of Object.keys(gsData)) {
    const gs = gsData[gsId];

    const geom = new THREE.SphereGeometry(65 * KM_SCALE, 16, 16);
    const mat = new THREE.MeshStandardMaterial({ color: 0xff3333 }); // red default
    const mesh = new THREE.Mesh(geom, mat);

    // Place on Earth surface in ECEF and attach to earthGroup so it rotates visually
    const p = latLonToECEF(gs.lat_deg, gs.lon_deg, gs.alt_km);
    p.multiplyScalar(1.003); // avoid z-fighting
    mesh.position.copy(p);

    mesh.userData = { type: "gs", gsId, name: gs.name };
    earthGroup.add(mesh);
    gsMeshes.set(gsId, mesh);
  }
}

function rebuildCoverageMesh() {
  // Remove old mesh if present
  if (coverageMesh) {
    earthGroup.remove(coverageMesh);
    coverageMesh.geometry.dispose();
    coverageMesh.material.dispose();
    coverageMesh = null;
  }

  if (!coverageEnabled || !coverageData) return;

  const latBins = coverageData.lat_bins;
  const lonBins = coverageData.lon_bins;
  const hits = coverageData.hits;

  // Find max hits for normalization
  let maxHits = 1;
  for (let i = 0; i < hits.length; i++) {
    for (let j = 0; j < hits[i].length; j++) {
      if (hits[i][j] > maxHits) maxHits = hits[i][j];
    }
  }

  const positions = [];
  const colors = [];

  const r = (EARTH_RADIUS_KM * KM_SCALE) * 1.003;  // slight lift

  for (let i = 0; i < latBins.length; i++) {
    const lat = latBins[i] * Math.PI / 180;
    const clat = Math.cos(lat);
    const slat = Math.sin(lat);

    for (let j = 0; j < lonBins.length; j++) {
      const h = hits[i][j];
      if (h <= 0) continue;

      const lon = lonBins[j] * Math.PI / 180;
      const clon = Math.cos(lon);
      const slon = Math.sin(lon);

      const x = r * clat * clon;
      const y = r * clat * slon;
      const z = r * slat;

      positions.push(x, y, z);

      // normalize and apply a tiny gamma to boost contrast
      let t = h / maxHits;
      t = Math.pow(t, 0.6);

      const c = heatColor(t);
      colors.push(c.r, c.g, c.b);
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geom.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size: 5,
    vertexColors: true,
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
  });

  coverageMesh = new THREE.Points(geom, mat);
  coverageMesh.userData = { type: "coverage" };
  earthGroup.add(coverageMesh);
}

function buildCoveragePointCloud() {
  if (!coverageData) return;

  const latBins = coverageData.lat_bins;
  const lonBins = coverageData.lon_bins;
  const hits = coverageData.hits;

  // Find max hits for normalization
  let maxHits = 1;
  for (let i = 0; i < hits.length; i++) {
    for (let j = 0; j < hits[i].length; j++) {
      if (hits[i][j] > maxHits) maxHits = hits[i][j];
    }
  }

  const positions = [];
  const colors = [];

  // Put points slightly above Earth surface to avoid z-fighting
  const r = (EARTH_RADIUS_KM * KM_SCALE) * 1.002;

  for (let i = 0; i < latBins.length; i++) {
    const lat = latBins[i] * Math.PI / 180;
    const clat = Math.cos(lat);
    const slat = Math.sin(lat);

    for (let j = 0; j < lonBins.length; j++) {
      const h = hits[i][j];
      if (h <= 0) continue;

      const lon = lonBins[j] * Math.PI / 180;
      const clon = Math.cos(lon);
      const slon = Math.sin(lon);

      const x = r * clat * clon;
      const y = r * clat * slon;
      const z = r * slat;

      positions.push(x, y, z);

      // intensity 0..1
      const intensity = h / maxHits;

      // Simple heat: dark -> bright (no fancy colormap)
      const c = 0.15 + 0.85 * intensity;
      colors.push(c, c, c); // grayscale; can upgrade later
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geom.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

  const mat = new THREE.PointsMaterial({
    size: 4,
    vertexColors: true,
    transparent: true,
    opacity: 0.85,
    depthWrite: false
  });

  const points = new THREE.Points(geom, mat);

  // Attach to earthGroup so it rotates with Earth
  earthGroup.add(points);

  coverageMesh = points;
}

function setEarthRotationForTime(t_s) {
  // Rotate Earth group about +Z to simulate Earth rotation (ECEF-like)
  // This is a visualization alignment; your visibility truth comes from log.
  earthGroup.rotation.z = OMEGA_EARTH_RAD_S * t_s;
}

function gsCanSeeSat(gsId, satId, i) {
  const key = `${gsId}|${satId}`;
  const series = visData[key];
  return series ? series[i] === 1 : false;
}

function gsSeesAny(gsId, i) {
  for (const satId of Object.keys(samplesBySat)) {
    if (gsCanSeeSat(gsId, satId, i)) return true;
  }
  return false;
}

function getVisibleSatsForGs(gsId, i) {
  const visible = [];
  for (const satId of Object.keys(samplesBySat)) {
    if (gsCanSeeSat(gsId, satId, i)) visible.push(satId);
  }
  return visible;
}

function updateHud() {
  selectedSatEl.textContent = `Satellite: ${selectedSatId ?? "(none)"}`;
  selectedGsEl.textContent = `Ground Station: ${selectedGsId ?? "(none)"}`;
  visModeLabel.textContent = selectedSatId ? `Mode: ${selectedSatId}` : "Mode: ANY satellite";
  followBtn.textContent = `Follow: ${followEnabled ? "ON" : "OFF"}`;

  // Visibility list content
  if (selectedGsId) {
    const sats = getVisibleSatsForGs(selectedGsId, frameIndex);
    if (sats.length === 0) {
      visList.textContent = "No satellites visible right now.";
    } else {
      visList.innerHTML = `Visible now:<br>${sats.map(s => `• ${s}`).join("<br>")}`;
    }
  } else {
    visList.textContent = "Click a ground station to see which satellites it can see.";
  }
}

function updateFrame(i) {
  const t = times[i] ?? 0;
  timeLabel.textContent = `t=${Math.round(t)}s`;
  timeSlider.value = String(i);

  // Rotate Earth + stations
  setEarthRotationForTime(t);

  // Satellite marker + trail
  const trailLen = 300;
  const start = Math.max(0, i - trailLen);

  for (const [sid, mesh] of satMeshes.entries()) {
    const samples = samplesBySat[sid];
    const r = samples[i];
    if (!r) continue;
    const [x, y, z] = r;
    mesh.position.set(x * KM_SCALE, y * KM_SCALE, z * KM_SCALE);

    // Trail
    const pts = [];
    for (let k = start; k <= i; k++) {
      const rr = samples[k];
      if (!rr) continue;
      pts.push(rr[0] * KM_SCALE, rr[1] * KM_SCALE, rr[2] * KM_SCALE);
    }
    const arr = new Float32Array(pts);
    const line = satTrails.get(sid);
    line.geometry.dispose();
    line.geometry = new THREE.BufferGeometry();
    line.geometry.setAttribute("position", new THREE.BufferAttribute(arr, 3));
  }

  // Ground station coloring:
  // - If a satellite is selected: GS green only when it sees that satellite
  // - Otherwise: GS green when it sees ANY sat
  for (const [gsId, mesh] of gsMeshes.entries()) {
    const canSee = selectedSatId
      ? gsCanSeeSat(gsId, selectedSatId, i)
      : gsSeesAny(gsId, i);
    mesh.material.color.set(canSee ? 0x33ff66 : 0xff3333);
  }

  updateHud();
}

// Raycasting for click selection
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.1;
raycaster.params.Line.threshold = 0.1;
const mouse = new THREE.Vector2();

function onClick(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);

  // We want clickable: sat markers + ground station markers
  const clickable = [
    ...Array.from(satMeshes.values()),
    ...Array.from(gsMeshes.values()),
  ];

  const hits = raycaster.intersectObjects(clickable, false);
  if (hits.length === 0) return;

  const obj = hits[0].object;
  const ud = obj.userData || {};

  if (ud.type === "sat") {
    selectedSatId = ud.sid;
    followEnabled = true;
    followBtn.textContent = "Follow: ON";
    satSelect.value = ud.sid;  // sync dropdown
  } else if (ud.type === "gs") {
    selectedGsId = ud.gsId;
  }

  updateFrame(frameIndex);
}

function onMouseMove(event) {
  if (!coverageData || !earthMesh) return;

  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);

  // Intersect Earth mesh (it sits inside earthGroup)
  const hits = raycaster.intersectObject(earthMesh, true);
  if (hits.length === 0) return;

  // Intersection point is in world coords; convert to earthGroup local coords
  const pWorld = hits[0].point.clone();
  const pLocal = earthGroup.worldToLocal(pWorld);

  const { lat, lon } = vecToLatLonDeg(pLocal);

  const latBins = coverageData.lat_bins;
  const lonBins = coverageData.lon_bins;

  const i = nearestIndex(latBins, lat);
  const j = nearestIndex(lonBins, lon);

  // Only update HUD if cell changed (avoid spam)
  if (lastHover.i === i && lastHover.j === j) return;

  lastHover = { i, j, lat, lon };

  const hitCount = coverageData.hits?.[i]?.[j] ?? 0;
  const stats = coverageData.revisit_stats?.[i]?.[j] ?? null;

  let statsText = "No revisit stats (not revisited yet).";
  if (stats) {
    const count = stats.count;
    const minS = stats.min_s;
    const meanS = stats.mean_s;
    const maxS = stats.max_s;
    statsText =
      `Revisit samples: ${count}<br>` +
      `Min: ${minS.toFixed(1)}s<br>` +
      `Mean: ${meanS.toFixed(1)}s<br>` +
      `Max: ${maxS.toFixed(1)}s`;
  }

  hoverInfo.innerHTML =
    `Lat: ${lat.toFixed(2)}°, Lon: ${lon.toFixed(2)}°<br>` +
    `Cell center: (${latBins[i].toFixed(2)}°, ${lonBins[j].toFixed(2)}°)<br>` +
    `Hits: ${hitCount}<br>` +
    statsText;
}

function animate() {
  requestAnimationFrame(animate);

  // Camera follow behavior
  if (followEnabled && selectedSatId && satMeshes.has(selectedSatId)) {
    const targetMesh = satMeshes.get(selectedSatId);
    const targetPos = targetMesh.position.clone();

    // Smoothly move controls target to satellite
    controls.target.lerp(targetPos, 0.1);

    // Get current distance to check for user zoom input
    const currentDist = camera.position.distanceTo(targetPos);
    
    // Default distance from slider
    const sliderDist = followDistance * EARTH_RADIUS_KM * KM_SCALE * 0.35;
    
    // If user hasn't set custom zoom, initialize it
    if (userZoomDistance === null) {
      userZoomDistance = sliderDist;
    }
    
    // Detect if user is zooming (distance changed by more than 5% from our target)
    const distDiff = Math.abs(currentDist - userZoomDistance);
    if (distDiff > userZoomDistance * 0.05) {
      // User is actively zooming - update our target distance
      userZoomDistance = currentDist;
    }
    
    // Calculate camera position maintaining user's zoom distance
    const toSat = targetPos.clone().normalize();
    const offsetDir = toSat.clone().multiplyScalar(userZoomDistance);
    const desiredCamPos = targetPos.clone().add(offsetDir);

    // Very light lerp to smooth movement without fighting user input
    camera.position.lerp(desiredCamPos, 0.05);

    controls.update();
  } else {
    // Reset user zoom when exiting follow mode
    userZoomDistance = null;
    controls.update();
  }

  if (isPlaying && times.length > 0) {
    playbackFrameCounter++;
    if (playbackFrameCounter >= 3) {  // advance every 3rd frame (~20 fps instead of 60)
      playbackFrameCounter = 0;
      frameIndex = (frameIndex + 1) % times.length;
      updateFrame(frameIndex);
    }
  }

  // if (cloudSphere) cloudSphere.rotation.y += 0.00015;

  renderer.render(scene, camera);
}

(async function main() {
  await loadBundle();
  initThree();
  createSatObjects();
  createGroundStations();

  // ensureCloudSphere();
  // addAtmosphereGlow();
  ensureNightLightsSphere();

  await loadCoverage(coverageSelect.value);

  renderer.domElement.addEventListener("click", onClick);
  renderer.domElement.addEventListener("mousemove", onMouseMove);

  frameIndex = 0;
  updateFrame(frameIndex);
  animate();
})();
