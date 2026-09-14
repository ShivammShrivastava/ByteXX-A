import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

/**
 * EarthCanvas
 * Renders a high-fidelity 3D rotating particle Earth with atmospheric halo and orbital tracks.
 * Interpolates camera and earth scale seamlessly on scroll to produce the sunset horizon sequence (Images 1 -> 5).
 */
export default function EarthCanvas({ scrollProgress = 0 }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const animFrameRef = useRef(null);
  const globeGroupRef = useRef(null);
  const cameraRef = useRef(null);
  const satellitesGroupRef = useRef(null);
  const atmospheresRef = useRef({ blueHalo: null, amberHalo: null });

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    // 1. Scene & Camera setup
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 0, 7.5);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    rendererRef.current = renderer;
    container.appendChild(renderer.domElement);

    // 2. Starfield background
    const starCount = 1800;
    const starGeo = new THREE.BufferGeometry();
    const starPos = new Float32Array(starCount * 3);
    const starColors = new Float32Array(starCount * 3);

    for (let i = 0; i < starCount; i++) {
      const r = THREE.MathUtils.randFloat(30, 80);
      const theta = THREE.MathUtils.randFloat(0, Math.PI * 2);
      const phi = THREE.MathUtils.randFloat(0, Math.PI);
      starPos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      starPos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      starPos[i * 3 + 2] = -Math.abs(r * Math.cos(phi));

      const colVal = THREE.MathUtils.randFloat(0.6, 1.0);
      const isBlueTint = Math.random() > 0.7;
      starColors[i * 3] = isBlueTint ? 0.7 * colVal : colVal;
      starColors[i * 3 + 1] = isBlueTint ? 0.85 * colVal : colVal;
      starColors[i * 3 + 2] = colVal;
    }
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    starGeo.setAttribute('color', new THREE.BufferAttribute(starColors, 3));
    const starMat = new THREE.PointsMaterial({
      size: 0.6,
      vertexColors: true,
      transparent: true,
      opacity: 0.85
    });
    const starField = new THREE.Points(starGeo, starMat);
    scene.add(starField);

    // 3. Globe Container (Rotates on axis)
    const globeGroup = new THREE.Group();
    globeGroup.position.set(2.2, 0, 0); // Positioned to the right as in Image 1
    globeGroup.rotation.z = 0.23; // Realistic 23.4° axial tilt
    globeGroupRef.current = globeGroup;
    scene.add(globeGroup);

    // 4. Continents & Particle Sphere
    // Generate geographical dots with realistic continental density
    const globeRadius = 2.8;
    const particleCount = 16000;
    const particleGeo = new THREE.BufferGeometry();
    const particlePositions = new Float32Array(particleCount * 3);
    const particleColors = new Float32Array(particleCount * 3);
    const particleSizes = new Float32Array(particleCount);

    // Approximate continent clusters: North America, South America, Eurasia, Africa, Australia
    const isLand = (lat, lon) => {
      // Eurasia & India
      if (lat >= 5 && lat <= 70 && lon >= -10 && lon <= 150) {
        if (lat >= 8 && lat <= 35 && lon >= 68 && lon <= 90) return true; // India / South Asia
        if (lat >= 30 && lat <= 65 && lon >= 10 && lon <= 130) return true; // Europe / Asia
        if (lat < 25 && lon > 40 && lon < 60) return false; // Arabian Sea
        return true;
      }
      // Africa
      if (lat >= -35 && lat <= 36 && lon >= -18 && lon <= 52) return true;
      // North America
      if (lat >= 15 && lat <= 72 && lon >= -168 && lon <= -52) return true;
      // South America
      if (lat >= -55 && lat <= 12 && lon >= -82 && lon <= -34) return true;
      // Australia
      if (lat >= -44 && lat <= -10 && lon >= 112 && lon <= 154) return true;
      // Ocean sparse points
      return Math.random() < 0.08;
    };

    let pIdx = 0;
    const goldenRatio = (1 + Math.sqrt(5)) / 2;

    for (let i = 0; i < particleCount * 2 && pIdx < particleCount; i++) {
      const theta = 2 * Math.PI * i / goldenRatio;
      const phi = Math.acos(1 - 2 * (i + 0.5) / (particleCount * 2));
      const lat = (Math.PI / 2 - phi) * (180 / Math.PI);
      const lon = (theta % (2 * Math.PI) - Math.PI) * (180 / Math.PI);

      const inLand = isLand(lat, lon);
      if (inLand || Math.random() < 0.12) {
        const rad = globeRadius + (inLand ? 0.01 : 0);
        const x = rad * Math.sin(phi) * Math.cos(theta);
        const y = rad * Math.cos(phi);
        const z = rad * Math.sin(phi) * Math.sin(theta);

        particlePositions[pIdx * 3] = x;
        particlePositions[pIdx * 3 + 1] = y;
        particlePositions[pIdx * 3 + 2] = z;

        if (inLand) {
          // Luminous white-cyan dots for landmasses
          const bright = THREE.MathUtils.randFloat(0.75, 1.0);
          particleColors[pIdx * 3] = 0.7 * bright;
          particleColors[pIdx * 3 + 1] = 0.9 * bright;
          particleColors[pIdx * 3 + 2] = 1.0 * bright;
          particleSizes[pIdx] = THREE.MathUtils.randFloat(0.04, 0.065);
        } else {
          // Subtle ocean grid points
          particleColors[pIdx * 3] = 0.1;
          particleColors[pIdx * 3 + 1] = 0.3;
          particleColors[pIdx * 3 + 2] = 0.55;
          particleSizes[pIdx] = 0.02;
        }
        pIdx++;
      }
    }

    particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePositions.slice(0, pIdx * 3), 3));
    particleGeo.setAttribute('color', new THREE.BufferAttribute(particleColors.slice(0, pIdx * 3), 3));
    particleGeo.setAttribute('size', new THREE.BufferAttribute(particleSizes.slice(0, pIdx), 1));

    // High performance point sprite material
    const particleMat = new THREE.PointsMaterial({
      size: 0.05,
      vertexColors: true,
      transparent: true,
      opacity: 0.92,
      blending: THREE.AdditiveBlending
    });
    const globePoints = new THREE.Points(particleGeo, particleMat);
    globeGroup.add(globePoints);

    // 5. Dark Inner Occlusion Sphere (so points on the back side are softly occluded)
    const occludeGeo = new THREE.SphereGeometry(globeRadius * 0.985, 48, 48);
    const occludeMat = new THREE.MeshBasicMaterial({
      color: 0x02040b,
      transparent: true,
      opacity: 0.94
    });
    const occludeMesh = new THREE.Mesh(occludeGeo, occludeMat);
    globeGroup.add(occludeMesh);

    // 6. Glowing Atmospheric Halo Layers (Matches Image 1: Gold rim on top, Electric Blue below)
    const atmosphereGeo = new THREE.SphereGeometry(globeRadius * 1.14, 48, 48);
    
    // Custom atmosphere shader
    const atmosphereShaderMat = new THREE.ShaderMaterial({
      vertexShader: `
        varying vec3 vNormal;
        varying vec3 vPosition;
        void main() {
          vNormal = normalize(normalMatrix * normal);
          vPosition = (modelViewMatrix * vec4(position, 1.0)).xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec3 vNormal;
        varying vec3 vPosition;
        uniform vec3 uBlueColor;
        uniform vec3 uAmberColor;
        void main() {
          vec3 viewDir = normalize(-vPosition);
          float fresnel = 1.0 - max(0.0, dot(viewDir, vNormal));
          fresnel = pow(fresnel, 2.6);

          // Top limb is warmer amber (Image 1 top rim), bottom limb is deep electric blue
          float verticalFactor = clamp((vNormal.y + 0.4) * 0.8, 0.0, 1.0);
          vec3 atmosphereColor = mix(uBlueColor, uAmberColor, pow(verticalFactor, 2.2));

          gl_FragColor = vec4(atmosphereColor, fresnel * 0.95);
        }
      `,
      uniforms: {
        uBlueColor: { value: new THREE.Color(0x0088ff) },
        uAmberColor: { value: new THREE.Color(0xff8822) }
      },
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
      transparent: true,
      depthWrite: false
    });
    const atmosphereMesh = new THREE.Mesh(atmosphereGeo, atmosphereShaderMat);
    globeGroup.add(atmosphereMesh);
    atmospheresRef.current.blueHalo = atmosphereMesh;

    // 7. Orbital Track Arcs & Remote Sensing Satellites
    const satellitesGroup = new THREE.Group();
    satellitesGroupRef.current = satellitesGroup;
    globeGroup.add(satellitesGroup);

    // Create 3 prominent orbital rings like in Image 1
    const createOrbit = (radius, inclination, rotationY, colorHex, satName) => {
      const points = [];
      const segments = 120;
      for (let i = 0; i <= segments; i++) {
        const theta = (i / segments) * Math.PI * 2;
        points.push(new THREE.Vector3(radius * Math.cos(theta), 0, radius * Math.sin(theta)));
      }
      const orbitGeo = new THREE.BufferGeometry().setFromPoints(points);
      const orbitMat = new THREE.LineBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity: 0.55,
        blending: THREE.AdditiveBlending
      });
      const orbitLine = new THREE.Line(orbitGeo, orbitMat);
      orbitLine.rotation.x = inclination;
      orbitLine.rotation.y = rotationY;
      satellitesGroup.add(orbitLine);

      // Satellite node
      const satGeo = new THREE.SphereGeometry(0.065, 12, 12);
      const satMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const satMesh = new THREE.Mesh(satGeo, satMat);
      
      // Satellite glow halo
      const satHaloGeo = new THREE.SphereGeometry(0.14, 12, 12);
      const satHaloMat = new THREE.MeshBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity: 0.7,
        blending: THREE.AdditiveBlending
      });
      const satHalo = new THREE.Mesh(satHaloGeo, satHaloMat);
      satMesh.add(satHalo);

      satellitesGroup.add(satMesh);

      return { line: orbitLine, satMesh, radius, speed: 0.6 + Math.random() * 0.4, inclination, rotationY, name: satName };
    };

    const orbits = [
      createOrbit(globeRadius * 1.25, 0.45, 0.2, 0xffaa33, 'Cartosat-2S'),
      createOrbit(globeRadius * 1.35, -0.6, -0.4, 0x00ddff, 'RISAT-SAR'),
      createOrbit(globeRadius * 1.18, 1.1, 0.8, 0x44aaff, 'Sentinel-2A')
    ];

    // 8. Ground station / target city markers (Pulsing orange dots like in Image 1)
    const stations = [
      { lat: 28.6, lon: 77.2, label: 'ISRO / SAC' },
      { lat: 40.7, lon: -74.0, label: 'USA' },
      { lat: 51.5, lon: -0.1, label: 'UK' },
      { lat: 48.8, lon: 2.3, label: 'FRANCE' },
      { lat: 35.6, lon: 139.6, label: 'JAPAN' }
    ];

    const markersGroup = new THREE.Group();
    stations.forEach((st) => {
      const phi = (90 - st.lat) * (Math.PI / 180);
      const theta = (st.lon + 180) * (Math.PI / 180);
      const r = globeRadius * 1.01;
      const x = -(r * Math.sin(phi) * Math.cos(theta));
      const z = (r * Math.sin(phi) * Math.sin(theta));
      const y = (r * Math.cos(phi));

      const markerGeo = new THREE.SphereGeometry(0.045, 8, 8);
      const markerMat = new THREE.MeshBasicMaterial({ color: 0xffaa22 });
      const marker = new THREE.Mesh(markerGeo, markerMat);
      marker.position.set(x, y, z);

      // Outer ring
      const ringGeo = new THREE.RingGeometry(0.05, 0.08, 16);
      const ringMat = new THREE.MeshBasicMaterial({
        color: 0xffaa22,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.6
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.position.set(x, y, z);
      ring.lookAt(x * 2, y * 2, z * 2);

      marker.add(ring);
      markersGroup.add(marker);
    });
    globeGroup.add(markersGroup);

    // 9. Interactive mouse parallax
    let mouseX = 0;
    let mouseY = 0;
    const handleMouseMove = (e) => {
      mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
      mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
    };
    window.addEventListener('mousemove', handleMouseMove);

    // Resize handler
    const handleResize = () => {
      if (!container || !camera || !renderer) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    // 10. Animation Loop
    let clock = new THREE.Clock();

    const animate = () => {
      animFrameRef.current = requestAnimationFrame(animate);
      const delta = clock.getDelta();
      const elapsed = clock.getElapsedTime();

      // Autonomous rotation of the Earth on its axis
      if (globeGroup) {
        globeGroup.rotation.y += delta * 0.16;
        // Subtle mouse tilt response
        globeGroup.rotation.x = 0.23 + mouseY * 0.08;
      }

      // Animate satellites along orbits
      orbits.forEach((orb) => {
        const t = elapsed * orb.speed;
        const x = orb.radius * Math.cos(t);
        const z = orb.radius * Math.sin(t);
        
        // Transform based on orbit orientation
        const satPos = new THREE.Vector3(x, 0, z);
        satPos.applyAxisAngle(new THREE.Vector3(1, 0, 0), orb.inclination);
        satPos.applyAxisAngle(new THREE.Vector3(0, 1, 0), orb.rotationY);
        orb.satMesh.position.copy(satPos);
      });

      renderer.render(scene, camera);
    };

    animate();

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('resize', handleResize);
      if (container && renderer.domElement) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  // Update scene based on scrollProgress (Interpolates Images 1 -> 2 -> 3 -> 4 -> 5)
  useEffect(() => {
    if (!globeGroupRef.current || !cameraRef.current) return;
    const globe = globeGroupRef.current;
    const camera = cameraRef.current;
    const p = Math.min(Math.max(scrollProgress, 0), 1);

    // Scroll Interpolation Sequence:
    // Stage 1 (p = 0): Image 1 - Full rotating Earth on the right
    // Stage 2 (p = 0.25 - 0.5): Image 2 & 3 - Earth zooms in, moves towards center-right/bottom
    // Stage 3 (p = 0.5 - 0.8): Image 4 - Curving horizontal atmospheric limb across bottom
    // Stage 4 (p = 0.8 - 1.0): Image 5 - Camera submerges into daylight atmospheric glow
    
    // Position X: 2.2 -> 1.0 -> 0.0
    globe.position.x = THREE.MathUtils.lerp(2.2, 0.0, Math.pow(p, 0.8));
    
    // Position Y: 0 -> -1.4 -> -5.8 (Dropping the planet center down so only the top curve/horizon remains)
    globe.position.y = THREE.MathUtils.lerp(0.0, -5.6, Math.pow(p, 1.2));

    // Scale / Camera distance: Zooms dramatically into the limb
    const currentScale = THREE.MathUtils.lerp(1.0, 2.7, Math.pow(p, 1.1));
    globe.scale.set(currentScale, currentScale, currentScale);

    // Camera Z distance: 7.5 down to 5.2
    camera.position.z = THREE.MathUtils.lerp(7.5, 4.8, p);
    camera.position.y = THREE.MathUtils.lerp(0, 0.8, p);

    // Atmosphere color shift during sunset transition
    if (atmospheresRef.current.blueHalo) {
      const mat = atmospheresRef.current.blueHalo.material;
      if (mat.uniforms) {
        // Deepening twilight into intense cyan daylight at the bottom
        const blueCol = new THREE.Color().lerpColors(
          new THREE.Color(0x0088ff),
          new THREE.Color(0x38bdf8),
          p
        );
        mat.uniforms.uBlueColor.value = blueCol;
      }
    }
  }, [scrollProgress]);

  return (
    <div 
      ref={mountRef} 
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        overflow: 'hidden',
        pointerEvents: 'none',
        zIndex: 1
      }}
    />
  );
}
