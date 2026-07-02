import React from 'react';

const STATUS = {
  landed: 'var(--status-landed, #22C55E)',
  partial: 'var(--status-partial, #F59E0B)',
  missed: 'var(--status-missed, #EF4444)',
  unmeasured: 'var(--status-unmeasured, #9CA3AF)',
};

/** 10px status dot (roofline landed / partial / missed / unmeasured), optional label. */
export function StatusDot({ status = 'landed', label, style }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontSize: 13, fontFamily: 'var(--font-sans, Helvetica, Arial, sans-serif)', color: 'var(--gray-strong, #374151)', ...style }}>
      <span style={{ width: 10, height: 10, borderRadius: '50%', background: STATUS[status] || STATUS.landed, flex: 'none' }}></span>
      {label}
    </span>
  );
}
