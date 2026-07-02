import React from 'react';

const TONES = {
  neutral: { bg: 'var(--gray-tint, #F9FAFB)', border: 'var(--gray-soft, #9CA3AF)', heading: 'var(--gray-strong, #374151)' },
  blue:    { bg: 'var(--blue-tint, #EFF6FF)', border: 'var(--blue, #3B82F6)', heading: 'var(--blue-dark, #1D4ED8)' },
  green:   { bg: 'var(--green-tint, #F0FDF4)', border: 'var(--green, #22C55E)', heading: 'var(--green-dark, #15803D)' },
  amber:   { bg: 'var(--amber-tint, #FEF3E4)', border: 'var(--amber, #F59E0B)', heading: 'var(--amber-dark, #B45309)' },
  red:     { bg: 'var(--red-tint, #FEF2F2)', border: 'var(--red, #EF4444)', heading: 'var(--red-dark, #B91C1C)' },
  purple:  { bg: 'var(--purple-tint, #F5F3FF)', border: 'var(--purple, #8B5CF6)', heading: 'var(--purple-dark, #6D28D9)' },
};

/** Tinted concept panel: tint fill, 1.5px solid border, 10px radius, optional heading. */
export function TintPanel({ tone = 'neutral', heading, subheading, dashed = false, children, style }) {
  const t = TONES[tone] || TONES.neutral;
  return (
    <div style={{
      background: t.bg, border: `1.5px ${dashed ? 'dashed' : 'solid'} ${t.border}`,
      borderRadius: 'var(--radius-panel, 10px)', padding: '14px 16px', boxSizing: 'border-box',
      fontFamily: 'var(--font-sans, Helvetica, Arial, sans-serif)', fontSize: 13, color: 'var(--gray-strong, #374151)',
      ...style,
    }}>
      {heading && <div style={{ fontSize: 16, fontWeight: 700, color: t.heading }}>{heading}</div>}
      {subheading && <div style={{ fontSize: 13, marginTop: 4 }}>{subheading}</div>}
      {children}
    </div>
  );
}
