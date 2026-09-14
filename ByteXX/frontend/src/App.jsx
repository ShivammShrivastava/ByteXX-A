import React, { useState, useEffect, useRef } from 'react';
import EarthCanvas from './components/EarthCanvas';
import HeroSection from './components/HeroSection';
import SatQueryWorkspace from './components/SatQueryWorkspace';
import AuthModal from './components/AuthModal';
import { onAuthChange, logoutUser } from './firebase/firebaseConfig';
import './App.css';

export default function App() {
  const [user, setUser] = useState(null);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [scrollProgress, setScrollProgress] = useState(0);
  const workspaceRef = useRef(null);

  useEffect(() => {
    const unsubscribe = onAuthChange((currentUser) => {
      if (currentUser) {
        setUser(currentUser);
      } else {
        const hasVisited = sessionStorage.getItem('bytex_has_visited');
        if (!hasVisited) {
          setIsAuthModalOpen(true);
          sessionStorage.setItem('bytex_has_visited', 'true');
        }
      }
    });
    return () => unsubscribe();
  }, []);

  // Smooth scroll progress with longer transition distance
  useEffect(() => {
    let rafId = null;
    let currentProgress = 0;

    const handleScroll = () => {
      const scrollY = window.scrollY;
      // Use 2x viewport height for a much longer, smoother transition
      const transitionDistance = window.innerHeight * 2;
      const targetProgress = Math.min(Math.max(scrollY / transitionDistance, 0), 1);
      
      // Smooth lerp for buttery transition
      const lerp = (start, end, factor) => start + (end - start) * factor;
      
      const smoothUpdate = () => {
        currentProgress = lerp(currentProgress, targetProgress, 0.12);
        // Snap when close enough
        if (Math.abs(currentProgress - targetProgress) < 0.001) {
          currentProgress = targetProgress;
        }
        setScrollProgress(currentProgress);
        
        if (Math.abs(currentProgress - targetProgress) > 0.001) {
          rafId = requestAnimationFrame(smoothUpdate);
        }
      };

      cancelAnimationFrame(rafId);
      rafId = requestAnimationFrame(smoothUpdate);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', handleScroll);
      cancelAnimationFrame(rafId);
    };
  }, []);

  const handleScrollToWorkspace = () => {
    if (workspaceRef.current) {
      workspaceRef.current.scrollIntoView({ behavior: 'smooth' });
    } else {
      window.scrollTo({ top: window.innerHeight * 2.2, behavior: 'smooth' });
    }
  };

  const handleLogout = async () => {
    await logoutUser();
    setUser(null);
    setIsAuthModalOpen(true);
  };

  // Compute smooth opacity values
  const earthOpacity = scrollProgress > 0.85 ? Math.max(0, 1 - (scrollProgress - 0.85) * 6.67) : 1;
  const horizonOpacity = Math.pow(Math.min(scrollProgress * 1.5, 1), 1.2);
  const daylightOpacity = Math.min(Math.max((scrollProgress - 0.5) * 2, 0), 1);

  return (
    <div className="app-root">
      {/* 3D Earth - fades out smoothly */}
      <div className="earth-canvas-fixed-layer" style={{ opacity: earthOpacity }}>
        <EarthCanvas scrollProgress={scrollProgress} />
      </div>

      {/* Atmospheric horizon - fades in gradually */}
      <div className="horizon-atmospheric-gradient" style={{ opacity: horizonOpacity, pointerEvents: 'none' }} />

      {/* Daylight fill - appears in the second half of scroll */}
      <div className="daylight-white-fill" style={{ opacity: daylightOpacity, pointerEvents: 'none' }} />

      {/* Hero */}
      <section className="hero-scroll-section">
        <HeroSection user={user} onScrollDown={handleScrollToWorkspace} />
      </section>

      {/* Extended scroll buffer for smooth horizon transition */}
      <div className="horizon-transition-buffer" />

      {/* Workspace */}
      <section ref={workspaceRef} className="workspace-scroll-section">
        <SatQueryWorkspace user={user} onLogout={handleLogout} />
      </section>

      {/* Auth Modal */}
      <AuthModal 
        isOpen={isAuthModalOpen} 
        onClose={() => setIsAuthModalOpen(false)}
        onAuthSuccess={(authedUser) => setUser(authedUser)}
      />
    </div>
  );
}
