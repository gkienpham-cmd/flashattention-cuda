/**
 * White figure canvas with house-style title + optional gray subtitle.
 * @startingPoint section="Figure primitives" subtitle="Blank IEEE figure canvas (column or full width)" viewport="700x300"
 */
export interface FigureFrameProps {
  /** Figure title, 22px bold near-black */
  title: string;
  /** Optional gray subtitle line under the title */
  subtitle?: string;
  /** 'column' (600px), 'full' (760px), or a custom px number */
  width?: 'column' | 'full' | number;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}
export declare function FigureFrame(props: FigureFrameProps): JSX.Element;
