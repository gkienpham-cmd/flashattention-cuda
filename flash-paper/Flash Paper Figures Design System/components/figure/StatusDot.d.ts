/** 10px status dot (roofline landed / partial / missed / unmeasured), optional label. */
export interface StatusDotProps {
  status?: 'landed' | 'partial' | 'missed' | 'unmeasured';
  /** Optional 13px gray label after the dot (for legends) */
  label?: string;
  style?: React.CSSProperties;
}
export declare function StatusDot(props: StatusDotProps): JSX.Element;
