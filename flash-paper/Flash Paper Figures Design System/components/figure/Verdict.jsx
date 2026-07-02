import React from 'react';

/** List row with a unicode verdict mark: green check, red cross, or amber "partial". */
export function Verdict({ yes, partial = false, children, style }) {
  const mark = partial ? 'partial' : yes ? '✓' : '✗';
  const markColor = partial ? 'var(--verdict-partial, #B45309)' : yes ? 'var(--verdict-yes, #22C55E)' : 'var(--verdict-no, #EF4444)';
  const textColor = partial ? 'var(--gray-strong, #374151)' : yes ? '#166534' : '#991B1B';
  return (
    <div style={{
      display: 'flex', gap: 8, alignItems: 'baseline', fontSize: 13,
      fontFamily: 'var(--font-sans, Helvetica, Arial, sans-serif)', color: textColor, margin: '8px 0', ...style,
    }}>
      <span style={{ fontWeight: 700, color: markColor, flex: 'none', fontSize: partial ? 13 : 15 }}>{mark}</span>
      <span>{children}</span>
    </div>
  );
}
