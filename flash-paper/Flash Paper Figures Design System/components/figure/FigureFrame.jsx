import React from 'react';

/** White figure canvas with house-style title + optional gray subtitle. */
export function FigureFrame({ title, subtitle, width = 'column', children, style }) {
  const w = width === 'full' ? 760 : width === 'column' ? 600 : width;
  return (
    <div style={{
      width: w, background: 'var(--paper, #fff)', boxSizing: 'border-box',
      fontFamily: 'var(--font-sans, Helvetica, Arial, sans-serif)',
      padding: '20px 20px 24px', ...style,
    }}>
      <div style={{ fontSize: 'var(--fig-title-size, 22px)', fontWeight: 700, color: 'var(--fig-title, #2C2C2A)' }}>{title}</div>
      {subtitle && <div style={{ fontSize: 13, color: 'var(--fig-caption, #6B7280)', marginTop: 6 }}>{subtitle}</div>}
      <div style={{ marginTop: 16 }}>{children}</div>
    </div>
  );
}
