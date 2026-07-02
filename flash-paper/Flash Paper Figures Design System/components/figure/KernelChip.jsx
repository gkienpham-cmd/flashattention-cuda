import React from 'react';
import { StatusDot } from './StatusDot.jsx';

const PHASES = {
  purple: { bg: 'var(--purple-tint, #F5F3FF)', border: 'var(--purple-border, #C4B5FD)', label: 'var(--purple-dark, #6D28D9)' },
  blue:   { bg: 'var(--blue-tint, #EFF6FF)', border: 'var(--blue-border, #93C5FD)', label: 'var(--blue-dark, #1D4ED8)' },
  amber:  { bg: 'var(--amber-tint, #FEF3E4)', border: 'var(--amber-border, #FCD34D)', label: 'var(--amber-dark, #B45309)' },
  green:  { bg: 'var(--green-tint, #F0FDF4)', border: 'var(--green-border, #86EFAC)', label: 'var(--green-dark, #15803D)' },
};

/** Rounded kernel chip: bold version label + 13px takeaway + status dot at right. */
export function KernelChip({ version, takeaway, status = 'landed', phase = 'blue', width = 168, style }) {
  const p = PHASES[phase] || PHASES.blue;
  return (
    <div style={{
      width, boxSizing: 'border-box', background: p.bg, border: `1px solid ${p.border}`,
      borderRadius: 'var(--radius-chip, 8px)', padding: '8px 10px', position: 'relative',
      fontFamily: 'var(--font-sans, Helvetica, Arial, sans-serif)', ...style,
    }}>
      <div style={{ fontSize: 14, fontWeight: 700, color: p.label, paddingRight: 16 }}>{version}</div>
      <div style={{ fontSize: 13, color: 'var(--gray-strong, #374151)', marginTop: 2 }}>{takeaway}</div>
      <StatusDot status={status} style={{ position: 'absolute', top: 10, right: 10 }} />
    </div>
  );
}
