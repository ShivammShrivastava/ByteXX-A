import React, { useState } from 'react';
import {
  X,
  Mail,
  User,
  Calendar,
  BarChart2,
  LogOut,
  Shield,
  Cpu,
  Wifi,
  WifiOff,
  Pin,
  Clock,
  ChevronRight,
  Copy,
  Check,
} from 'lucide-react';
import './ProfilePanel.css';

export default function ProfilePanel({ user, recentsCount, pinnedCount, onLogout, onClose }) {
  const [copiedUid, setCopiedUid] = useState(false);

  const displayName = user?.displayName || user?.email?.split('@')[0] || 'Analyst';
  const email       = user?.email || 'analyst@bytex.ai';
  const isDemo      = !user?.email || user?.uid?.startsWith('user_') || user?.uid?.startsWith('demo');
  const avatarUrl   = user?.photoURL || `https://api.dicebear.com/7.x/bottts/svg?seed=${user?.uid || 'analyst'}`;

  const joinDate = (() => {
    if (user?.metadata?.creationTime) {
      return new Date(user.metadata.creationTime).toLocaleDateString('en-US', {
        month: 'long', day: 'numeric', year: 'numeric',
      });
    }
    return 'Today';
  })();

  const handleCopyUid = () => {
    if (user?.uid) {
      navigator.clipboard.writeText(user.uid).catch(() => {});
      setCopiedUid(true);
      setTimeout(() => setCopiedUid(false), 2000);
    }
  };

  return (
    <div className="profile-page-overlay" onClick={onClose}>
      <div className="profile-page" onClick={(e) => e.stopPropagation()}>

        {/* ── TOP BAR ── */}
        <div className="profile-page-topbar">
          <button className="profile-back-btn" onClick={onClose} aria-label="Close profile">
            <X size={20} />
          </button>
          <span className="profile-page-title">My Profile</span>
          <div style={{ width: 36 }} />
        </div>

        {/* ── SCROLLABLE BODY ── */}
        <div className="profile-page-body">

          {/* Hero Avatar Card */}
          <div className="profile-hero-card">
            <div className="profile-hero-glow" />
            <div className="profile-avatar-wrapper">
              <img src={avatarUrl} alt={displayName} className="profile-avatar-xl" />
              <div className={`profile-status-ring ${isDemo ? 'demo' : 'online'}`} />
            </div>
            <div className="profile-hero-text">
              <h1 className="profile-hero-name">{displayName}</h1>
              <span className={`profile-role-badge ${isDemo ? 'demo' : 'real'}`}>
                <Shield size={11} />
                {isDemo ? 'Demo Session' : 'Verified Analyst'}
              </span>
            </div>
          </div>

          {/* ── INFO SECTION ── */}
          <section className="profile-section">
            <span className="profile-section-label">Account Info</span>
            <div className="profile-info-list">

              <div className="profile-info-row">
                <div className="profile-info-icon-wrap blue">
                  <Mail size={15} />
                </div>
                <div className="profile-info-text">
                  <span className="profile-info-key">Email</span>
                  <span className="profile-info-val">{email}</span>
                </div>
              </div>

              <div className="profile-info-row" onClick={handleCopyUid} style={{ cursor: 'pointer' }}>
                <div className="profile-info-icon-wrap purple">
                  <User size={15} />
                </div>
                <div className="profile-info-text">
                  <span className="profile-info-key">User ID</span>
                  <span className="profile-info-val uid-mono">
                    {user?.uid ? user.uid.substring(0, 18) + '…' : '—'}
                  </span>
                </div>
                <div className="profile-copy-btn">
                  {copiedUid ? <Check size={14} className="copy-check" /> : <Copy size={14} />}
                </div>
              </div>

              <div className="profile-info-row">
                <div className="profile-info-icon-wrap green">
                  <Calendar size={15} />
                </div>
                <div className="profile-info-text">
                  <span className="profile-info-key">Member Since</span>
                  <span className="profile-info-val">{joinDate}</span>
                </div>
              </div>

              <div className="profile-info-row">
                <div className="profile-info-icon-wrap orange">
                  <Cpu size={15} />
                </div>
                <div className="profile-info-text">
                  <span className="profile-info-key">Active Model</span>
                  <span className="profile-info-val">Google Gemma 4 31B (OpenRouter)</span>
                </div>
              </div>

            </div>
          </section>

          {/* ── STATS SECTION ── */}
          <section className="profile-section">
            <span className="profile-section-label">Usage Stats</span>
            <div className="profile-stats-grid">
              <div className="profile-stat-card">
                <BarChart2 size={22} className="stat-card-icon blue" />
                <span className="stat-card-number">{recentsCount}</span>
                <span className="stat-card-label">Analyses Run</span>
              </div>
              <div className="profile-stat-card">
                <Pin size={22} className="stat-card-icon purple" />
                <span className="stat-card-number">{pinnedCount}</span>
                <span className="stat-card-label">Pinned</span>
              </div>
              <div className="profile-stat-card">
                <Clock size={22} className="stat-card-icon green" />
                <span className="stat-card-number">3</span>
                <span className="stat-card-label">Modalities</span>
              </div>
            </div>
          </section>

          {/* ── CONNECTION STATUS ── */}
          <section className="profile-section">
            <span className="profile-section-label">Connection</span>
            <div className="profile-connection-card">
              <div className={`conn-dot ${isDemo ? 'offline' : 'online'}`} />
              <div className="conn-text">
                <span className="conn-title">
                  {isDemo ? 'Local Demo Session' : 'Firebase RTDB Connected'}
                </span>
                <span className="conn-sub">
                  {isDemo
                    ? 'Sign in with Google for cloud query history'
                    : 'satellite-efa0a.firebaseio.com'}
                </span>
              </div>
              {isDemo
                ? <WifiOff size={18} className="conn-icon offline" />
                : <Wifi size={18} className="conn-icon online" />
              }
            </div>
          </section>

        </div>

        {/* ── BOTTOM LOGOUT BUTTON ── */}
        <div className="profile-page-footer">
          <button
            className="profile-logout-full-btn"
            onClick={() => { onLogout(); onClose(); }}
          >
            <LogOut size={18} />
            <span>Sign Out</span>
          </button>
        </div>

      </div>
    </div>
  );
}
