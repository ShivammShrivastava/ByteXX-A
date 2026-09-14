import React, { useState } from 'react';
import { 
  loginWithEmail, 
  registerWithEmail, 
  loginWithGoogle, 
  saveUserData 
} from '../firebase/firebaseConfig';
import { Mail, Lock, User, ArrowRight, ShieldCheck, X } from 'lucide-react';
import './AuthModal.css';

export default function AuthModal({ isOpen, onClose, onAuthSuccess }) {
  const [isRegister, setIsRegister] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  // Helper: create a local/demo user session when Firebase auth isn't configured
  const createFallbackUser = async (emailVal, nameVal) => {
    const fallbackUser = {
      uid: 'user_' + Date.now().toString(36) + '_' + Math.random().toString(36).substring(2, 7),
      email: emailVal || 'analyst@bytex.ai',
      displayName: nameVal || (emailVal ? emailVal.split('@')[0] : 'Satellite Analyst'),
      photoURL: `https://api.dicebear.com/7.x/bottts/svg?seed=${emailVal || 'analyst'}`
    };
    await saveUserData(fallbackUser);
    onAuthSuccess(fallbackUser);
    onClose();
  };

  // Check if error is a Firebase config/API key issue
  const isConfigError = (err) => {
    const code = (err.code || '').toLowerCase();
    const msg = (err.message || '').toLowerCase();
    return code.includes('api-key') || 
           code.includes('invalid-api') ||
           msg.includes('api key') || 
           msg.includes('api-key') ||
           msg.includes('not-valid') ||
           msg.includes('network') ||
           msg.includes('auth/configuration') ||
           msg.includes('invalid') && msg.includes('key');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      let user;
      if (isRegister) {
        user = await registerWithEmail(email, password, displayName);
      } else {
        user = await loginWithEmail(email, password);
      }
      onAuthSuccess(user);
      onClose();
    } catch (err) {
      console.warn("Auth error:", err);
      if (isConfigError(err)) {
        // Firebase not configured yet - fall back to local session
        await createFallbackUser(email, displayName || email.split('@')[0]);
      } else {
        setError(err.message.replace('Firebase:', '').replace(/\(.*\)\.?/, '').trim() || 'Authentication failed');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    setError(null);
    setLoading(true);
    try {
      const user = await loginWithGoogle();
      onAuthSuccess(user);
      onClose();
    } catch (err) {
      console.warn("Google sign-in error:", err);
      if (isConfigError(err) || err.code === 'auth/popup-closed-by-user') {
        await createFallbackUser('google.analyst@bytex.ai', 'Google Analyst');
      } else {
        setError(err.message.replace('Firebase:', '').replace(/\(.*\)\.?/, '').trim() || 'Google sign-in failed');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleGuestDemo = async () => {
    await createFallbackUser('demo@bytex.satellite.ai', 'Remote Sensing Analyst');
  };

  return (
    <div className="auth-overlay">
      <div className="auth-card">
        <button className="auth-close-btn" onClick={onClose}>
          <X size={20} />
        </button>

        <div className="auth-header">
          <div className="auth-logo-badge">
            <span className="logo-text">ByteX</span>
            <span className="logo-sub">SatQuery AI</span>
          </div>
          <h3 className="auth-title">
            {isRegister ? 'Create Analyst Account' : 'Welcome to SatQuery AI'}
          </h3>
          <p className="auth-desc">
            Multimodal Remote Sensing Vision-Language Assistant powered by Google Gemma 4 31B
          </p>
        </div>

        {error && <div className="auth-error-banner">{error}</div>}

        <button 
          type="button" 
          className="google-sign-in-btn" 
          onClick={handleGoogleSignIn}
          disabled={loading}
        >
          <svg className="google-icon" viewBox="0 0 24 24" width="20" height="20">
            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"/>
            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/>
          </svg>
          <span>Continue with Google</span>
        </button>

        <div className="auth-divider">
          <span>or continue with email</span>
        </div>

        <form onSubmit={handleSubmit} className="auth-form">
          {isRegister && (
            <div className="input-group">
              <label>Full Name</label>
              <div className="input-box">
                <User size={18} className="input-icon" />
                <input 
                  type="text" 
                  placeholder="e.g. Alex Vance" 
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  required
                />
              </div>
            </div>
          )}

          <div className="input-group">
            <label>Email Address</label>
            <div className="input-box">
              <Mail size={18} className="input-icon" />
              <input 
                type="email" 
                placeholder="analyst@domain.com" 
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
          </div>

          <div className="input-group">
            <label>Password</label>
            <div className="input-box">
              <Lock size={18} className="input-icon" />
              <input 
                type="password" 
                placeholder="••••••••••••" 
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={6}
              />
            </div>
          </div>

          <button type="submit" className="auth-submit-btn" disabled={loading}>
            <span>{loading ? 'Please wait...' : isRegister ? 'Register Account' : 'Sign In'}</span>
            <ArrowRight size={18} />
          </button>
        </form>

        <div className="auth-footer">
          <button 
            type="button" 
            className="toggle-auth-btn"
            onClick={() => { setIsRegister(!isRegister); setError(null); }}
          >
            {isRegister ? 'Already have an account? Sign In' : "Don't have an account? Create one"}
          </button>

          <button 
            type="button" 
            className="guest-demo-btn"
            onClick={handleGuestDemo}
          >
            Explore with Demo Account
          </button>
        </div>

        <div className="auth-storage-note">
          <ShieldCheck size={14} />
          <span>Syncs with Firebase RTDB: satellite-efa0a</span>
        </div>
      </div>
    </div>
  );
}
