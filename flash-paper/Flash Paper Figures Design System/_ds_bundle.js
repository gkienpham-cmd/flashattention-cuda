/* @ds-bundle: {"format":3,"namespace":"FlashPaperFiguresDesignSystem_3f7340","components":[{"name":"FigureFrame","sourcePath":"components/figure/FigureFrame.jsx"},{"name":"KernelChip","sourcePath":"components/figure/KernelChip.jsx"},{"name":"StatusDot","sourcePath":"components/figure/StatusDot.jsx"},{"name":"TintPanel","sourcePath":"components/figure/TintPanel.jsx"},{"name":"Verdict","sourcePath":"components/figure/Verdict.jsx"}],"sourceHashes":{"components/figure/FigureFrame.jsx":"5a915d99bd5e","components/figure/KernelChip.jsx":"92b8b7c6c53d","components/figure/StatusDot.jsx":"a91e74d930dc","components/figure/TintPanel.jsx":"71a836577419","components/figure/Verdict.jsx":"00e817eeb0df"},"inlinedExternals":[],"unexposedExports":[]} */

(() => {

const __ds_ns = (window.FlashPaperFiguresDesignSystem_3f7340 = window.FlashPaperFiguresDesignSystem_3f7340 || {});

const __ds_scope = {};

(__ds_ns.__errors = __ds_ns.__errors || []);

// components/figure/FigureFrame.jsx
try { (() => {
/** White figure canvas with house-style title + optional gray subtitle. */
function FigureFrame({
  title,
  subtitle,
  width = 'column',
  children,
  style
}) {
  const w = width === 'full' ? 760 : width === 'column' ? 600 : width;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      width: w,
      background: 'var(--paper, #fff)',
      boxSizing: 'border-box',
      fontFamily: 'var(--font-sans, Helvetica, Arial, sans-serif)',
      padding: '20px 20px 24px',
      ...style
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 'var(--fig-title-size, 22px)',
      fontWeight: 700,
      color: 'var(--fig-title, #2C2C2A)'
    }
  }, title), subtitle && /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 13,
      color: 'var(--fig-caption, #6B7280)',
      marginTop: 6
    }
  }, subtitle), /*#__PURE__*/React.createElement("div", {
    style: {
      marginTop: 16
    }
  }, children));
}
Object.assign(__ds_scope, { FigureFrame });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/figure/FigureFrame.jsx", error: String((e && e.message) || e) }); }

// components/figure/StatusDot.jsx
try { (() => {
const STATUS = {
  landed: 'var(--status-landed, #22C55E)',
  partial: 'var(--status-partial, #F59E0B)',
  missed: 'var(--status-missed, #EF4444)',
  unmeasured: 'var(--status-unmeasured, #9CA3AF)'
};

/** 10px status dot (roofline landed / partial / missed / unmeasured), optional label. */
function StatusDot({
  status = 'landed',
  label,
  style
}) {
  return /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'inline-flex',
      alignItems: 'center',
      gap: 8,
      fontSize: 13,
      fontFamily: 'var(--font-sans, Helvetica, Arial, sans-serif)',
      color: 'var(--gray-strong, #374151)',
      ...style
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      width: 10,
      height: 10,
      borderRadius: '50%',
      background: STATUS[status] || STATUS.landed,
      flex: 'none'
    }
  }), label);
}
Object.assign(__ds_scope, { StatusDot });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/figure/StatusDot.jsx", error: String((e && e.message) || e) }); }

// components/figure/KernelChip.jsx
try { (() => {
const PHASES = {
  purple: {
    bg: 'var(--purple-tint, #F5F3FF)',
    border: 'var(--purple-border, #C4B5FD)',
    label: 'var(--purple-dark, #6D28D9)'
  },
  blue: {
    bg: 'var(--blue-tint, #EFF6FF)',
    border: 'var(--blue-border, #93C5FD)',
    label: 'var(--blue-dark, #1D4ED8)'
  },
  amber: {
    bg: 'var(--amber-tint, #FEF3E4)',
    border: 'var(--amber-border, #FCD34D)',
    label: 'var(--amber-dark, #B45309)'
  },
  green: {
    bg: 'var(--green-tint, #F0FDF4)',
    border: 'var(--green-border, #86EFAC)',
    label: 'var(--green-dark, #15803D)'
  }
};

/** Rounded kernel chip: bold version label + 13px takeaway + status dot at right. */
function KernelChip({
  version,
  takeaway,
  status = 'landed',
  phase = 'blue',
  width = 168,
  style
}) {
  const p = PHASES[phase] || PHASES.blue;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      width,
      boxSizing: 'border-box',
      background: p.bg,
      border: `1px solid ${p.border}`,
      borderRadius: 'var(--radius-chip, 8px)',
      padding: '8px 10px',
      position: 'relative',
      fontFamily: 'var(--font-sans, Helvetica, Arial, sans-serif)',
      ...style
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 14,
      fontWeight: 700,
      color: p.label,
      paddingRight: 16
    }
  }, version), /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 13,
      color: 'var(--gray-strong, #374151)',
      marginTop: 2
    }
  }, takeaway), /*#__PURE__*/React.createElement(__ds_scope.StatusDot, {
    status: status,
    style: {
      position: 'absolute',
      top: 10,
      right: 10
    }
  }));
}
Object.assign(__ds_scope, { KernelChip });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/figure/KernelChip.jsx", error: String((e && e.message) || e) }); }

// components/figure/TintPanel.jsx
try { (() => {
const TONES = {
  neutral: {
    bg: 'var(--gray-tint, #F9FAFB)',
    border: 'var(--gray-soft, #9CA3AF)',
    heading: 'var(--gray-strong, #374151)'
  },
  blue: {
    bg: 'var(--blue-tint, #EFF6FF)',
    border: 'var(--blue, #3B82F6)',
    heading: 'var(--blue-dark, #1D4ED8)'
  },
  green: {
    bg: 'var(--green-tint, #F0FDF4)',
    border: 'var(--green, #22C55E)',
    heading: 'var(--green-dark, #15803D)'
  },
  amber: {
    bg: 'var(--amber-tint, #FEF3E4)',
    border: 'var(--amber, #F59E0B)',
    heading: 'var(--amber-dark, #B45309)'
  },
  red: {
    bg: 'var(--red-tint, #FEF2F2)',
    border: 'var(--red, #EF4444)',
    heading: 'var(--red-dark, #B91C1C)'
  },
  purple: {
    bg: 'var(--purple-tint, #F5F3FF)',
    border: 'var(--purple, #8B5CF6)',
    heading: 'var(--purple-dark, #6D28D9)'
  }
};

/** Tinted concept panel: tint fill, 1.5px solid border, 10px radius, optional heading. */
function TintPanel({
  tone = 'neutral',
  heading,
  subheading,
  dashed = false,
  children,
  style
}) {
  const t = TONES[tone] || TONES.neutral;
  return /*#__PURE__*/React.createElement("div", {
    style: {
      background: t.bg,
      border: `1.5px ${dashed ? 'dashed' : 'solid'} ${t.border}`,
      borderRadius: 'var(--radius-panel, 10px)',
      padding: '14px 16px',
      boxSizing: 'border-box',
      fontFamily: 'var(--font-sans, Helvetica, Arial, sans-serif)',
      fontSize: 13,
      color: 'var(--gray-strong, #374151)',
      ...style
    }
  }, heading && /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 16,
      fontWeight: 700,
      color: t.heading
    }
  }, heading), subheading && /*#__PURE__*/React.createElement("div", {
    style: {
      fontSize: 13,
      marginTop: 4
    }
  }, subheading), children);
}
Object.assign(__ds_scope, { TintPanel });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/figure/TintPanel.jsx", error: String((e && e.message) || e) }); }

// components/figure/Verdict.jsx
try { (() => {
/** List row with a unicode verdict mark: green check, red cross, or amber "partial". */
function Verdict({
  yes,
  partial = false,
  children,
  style
}) {
  const mark = partial ? 'partial' : yes ? '✓' : '✗';
  const markColor = partial ? 'var(--verdict-partial, #B45309)' : yes ? 'var(--verdict-yes, #22C55E)' : 'var(--verdict-no, #EF4444)';
  const textColor = partial ? 'var(--gray-strong, #374151)' : yes ? '#166534' : '#991B1B';
  return /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      gap: 8,
      alignItems: 'baseline',
      fontSize: 13,
      fontFamily: 'var(--font-sans, Helvetica, Arial, sans-serif)',
      color: textColor,
      margin: '8px 0',
      ...style
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      fontWeight: 700,
      color: markColor,
      flex: 'none',
      fontSize: partial ? 13 : 15
    }
  }, mark), /*#__PURE__*/React.createElement("span", null, children));
}
Object.assign(__ds_scope, { Verdict });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/figure/Verdict.jsx", error: String((e && e.message) || e) }); }

__ds_ns.FigureFrame = __ds_scope.FigureFrame;

__ds_ns.KernelChip = __ds_scope.KernelChip;

__ds_ns.StatusDot = __ds_scope.StatusDot;

__ds_ns.TintPanel = __ds_scope.TintPanel;

__ds_ns.Verdict = __ds_scope.Verdict;

})();
