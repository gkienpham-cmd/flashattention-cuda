/** List row with a unicode verdict mark: green check, red cross, or amber "partial". */
export interface VerdictProps {
  /** true → green ✓, false → red ✗ */
  yes?: boolean;
  /** Renders the amber word "partial" instead of a glyph */
  partial?: boolean;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}
export declare function Verdict(props: VerdictProps): JSX.Element;
