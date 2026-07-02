/** Rounded kernel chip: bold version label + 13px takeaway + status dot at right. */
export interface KernelChipProps {
  /** Bold label, e.g. "v6 split-KV" */
  version: string;
  /** 3–5 word takeaway line */
  takeaway: string;
  status?: 'landed' | 'partial' | 'missed' | 'unmeasured';
  /** Phase tint (v1–v12 arc: 1 purple, 2 blue, 3 amber, 4 green) */
  phase?: 'purple' | 'blue' | 'amber' | 'green';
  /** px width; default 168 */
  width?: number;
  style?: React.CSSProperties;
}
export declare function KernelChip(props: KernelChipProps): JSX.Element;
