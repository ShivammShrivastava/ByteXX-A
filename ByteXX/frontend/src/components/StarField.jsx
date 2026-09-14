import { useEffect, useRef } from 'react';

/**
 * StarField — animated canvas background
 * - 150 stars with visible twinkling (opacity + radius pulse)
 * - Frequent comets with white gradient tails, occasional double-spawns
 */
export default function StarField() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let animId;
    let width = 0, height = 0;

    // ─── Stars ───────────────────────────────────────────────────────────────
    const STAR_COUNT = 160;
    const stars = [];

    const initStars = () => {
      stars.length = 0;
      for (let i = 0; i < STAR_COUNT; i++) {
        const r = Math.random() * 1.4 + 0.4;   // base radius 0.4 – 1.8
        stars.push({
          x:         Math.random() * width,
          y:         Math.random() * height,
          baseR:     r,
          phase:     Math.random() * Math.PI * 2,   // independent phase per star
          speed:     Math.random() * 0.035 + 0.015, // faster twinkle (0.015 – 0.05)
          baseAlpha: Math.random() * 0.5 + 0.35,    // 0.35 – 0.85 base brightness
        });
      }
    };

    const drawStar = (s, t) => {
      const wave  = Math.sin(t * s.speed + s.phase);
      // Opacity oscillates ±0.38 around baseAlpha — very visible twinkle
      const alpha = Math.max(0.04, Math.min(1, s.baseAlpha + wave * 0.38));
      // Radius also pulses slightly (±25%)
      const r     = s.baseR * (0.75 + (wave + 1) * 0.125);

      // Soft glow around brighter stars
      if (alpha > 0.6 && r > 0.9) {
        const glow = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, r * 3.5);
        glow.addColorStop(0,   `rgba(255,255,255,${alpha * 0.35})`);
        glow.addColorStop(1,   `rgba(255,255,255,0)`);
        ctx.beginPath();
        ctx.arc(s.x, s.y, r * 3.5, 0, Math.PI * 2);
        ctx.fillStyle = glow;
        ctx.fill();
      }

      // Solid star core
      ctx.beginPath();
      ctx.arc(s.x, s.y, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255,255,255,${alpha})`;
      ctx.fill();
    };

    // ─── Comets ───────────────────────────────────────────────────────────────
    const comets = [];
    let cometTimer = 0;
    // Spawn a comet every 100–180 frames (~1.7–3s at 60 fps) — frequent!
    let nextCometIn = 60;

    const spawnComet = () => {
      const fromTop = Math.random() > 0.35;
      const x = fromTop ? Math.random() * width  * 0.9 : -30;
      const y = fromTop ? -30                           : Math.random() * height * 0.6;

      const angleDeg = Math.random() * 24 + 20;   // 20–44°
      const angleRad = angleDeg * (Math.PI / 180);
      const speed    = Math.random() * 4 + 4.5;   // 4.5 – 8.5 px/frame

      comets.push({
        x, y,
        vx:      Math.cos(angleRad) * speed,
        vy:      Math.sin(angleRad) * speed,
        tailLen: Math.random() * 110 + 70,         // 70–180 px tail
        life:    0,
        maxLife: Math.random() * 90 + 120,         // 120–210 frames
      });
    };

    const drawComet = (c) => {
      const fadeIn  = Math.min(1, c.life / 16);
      const fadeOut = Math.min(1, (c.maxLife - c.life) / 16);
      const alpha   = Math.min(fadeIn, fadeOut);
      if (alpha <= 0.01) return;

      const mag = Math.hypot(c.vx, c.vy);
      const nx  = c.vx / mag;
      const ny  = c.vy / mag;

      const tx = c.x - nx * c.tailLen;
      const ty = c.y - ny * c.tailLen;

      // Gradient tail
      const grad = ctx.createLinearGradient(tx, ty, c.x, c.y);
      grad.addColorStop(0,    `rgba(255,255,255,0)`);
      grad.addColorStop(0.5,  `rgba(210,235,255,${alpha * 0.28})`);
      grad.addColorStop(0.85, `rgba(230,245,255,${alpha * 0.65})`);
      grad.addColorStop(1,    `rgba(255,255,255,${alpha * 0.95})`);

      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(c.x, c.y);
      ctx.strokeStyle = grad;
      ctx.lineWidth   = 1.6;
      ctx.lineCap     = 'round';
      ctx.stroke();

      // Outer glow halo
      const haloR = 6;
      const halo  = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, haloR);
      halo.addColorStop(0,   `rgba(200,230,255,${alpha * 0.6})`);
      halo.addColorStop(1,   `rgba(255,255,255,0)`);
      ctx.beginPath();
      ctx.arc(c.x, c.y, haloR, 0, Math.PI * 2);
      ctx.fillStyle = halo;
      ctx.fill();

      // Bright solid head
      ctx.beginPath();
      ctx.arc(c.x, c.y, 1.8, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255,255,255,${alpha})`;
      ctx.fill();
    };

    // ─── Resize ───────────────────────────────────────────────────────────────
    const resize = () => {
      width  = canvas.width  = canvas.offsetWidth;
      height = canvas.height = canvas.offsetHeight;
    };

    // ─── Animation loop ───────────────────────────────────────────────────────
    let t = 0;

    const draw = () => {
      ctx.clearRect(0, 0, width, height);
      t++;

      // Stars
      for (const s of stars) drawStar(s, t);

      // Comet spawning — occasional double-spawn for drama
      cometTimer++;
      if (cometTimer >= nextCometIn) {
        spawnComet();
        if (Math.random() < 0.3) spawnComet(); // 30% chance of a second simultaneous comet
        cometTimer = 0;
        nextCometIn = Math.random() * 80 + 100; // 100–180 frames next interval
      }

      // Update & draw comets
      for (let i = comets.length - 1; i >= 0; i--) {
        const c = comets[i];
        c.x += c.vx;
        c.y += c.vy;
        c.life++;
        if (c.life > c.maxLife || c.x > width + 300 || c.y > height + 300) {
          comets.splice(i, 1);
          continue;
        }
        drawComet(c);
      }

      animId = requestAnimationFrame(draw);
    };

    // ─── Init ─────────────────────────────────────────────────────────────────
    resize();
    initStars();
    // Burst of 2 comets right away so the screen feels alive immediately
    setTimeout(() => { spawnComet(); spawnComet(); }, 600);
    draw();

    const ro = new ResizeObserver(() => { resize(); initStars(); });
    ro.observe(canvas);

    return () => {
      cancelAnimationFrame(animId);
      ro.disconnect();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      style={{
        position:      'absolute',
        inset:         0,
        width:         '100%',
        height:        '100%',
        pointerEvents: 'none',
        zIndex:        0,
        display:       'block',
      }}
    />
  );
}
