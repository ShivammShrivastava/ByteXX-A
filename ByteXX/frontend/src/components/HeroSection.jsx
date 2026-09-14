import React from 'react';
import { ChevronDown, ArrowDown } from 'lucide-react';
import './HeroSection.css';

export default function HeroSection({ user, onScrollDown }) {
  const displayName = user?.displayName || user?.email?.split('@')[0] || 'Explorer';

  return (
    <div className="hero-viewport">
      {/* Top Left: ByteX Website Name */}
      <header className="hero-header">
        <div className="hero-header-left">
          <div className="bytex-brand-topleft">
            <span className="bytex-name">ByteX</span>
          </div>
        </div>
        <div className="hero-header-right"></div>
      </header>

      {/* Main Hero Body: Only "hi again" and "userName" - big, elegant, minimalist */}
      <div className="hero-content-container">
        <div className="hero-left-col">
          <div className="hero-title-group">
            <h1 className="hero-greeting-line">hi again</h1>
            <h2 className="hero-username-line">{displayName}</h2>
          </div>

          <button className="hero-primary-cta" onClick={onScrollDown}>
            <span>Launch SatQuery AI</span>
            <ArrowDown size={18} className="cta-bounce" />
          </button>
        </div>
      </div>

      {/* Bottom Scroll Prompter */}
      <div className="scroll-horizon-prompter" onClick={onScrollDown}>
        <div className="mouse-scroll-wheel">
          <div className="mouse-wheel-dot"></div>
        </div>
        <ChevronDown size={18} className="chevron-pulse" />
      </div>
    </div>
  );
}
