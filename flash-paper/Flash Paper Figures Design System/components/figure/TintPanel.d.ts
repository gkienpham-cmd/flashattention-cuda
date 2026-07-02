/** Tinted concept panel: tint fill, 1.5px solid border, 10px radius, optional heading. */
export interface TintPanelProps {
  /** Panel tone; maps to tint fill + border + heading color */
  tone?: 'neutral' | 'blue' | 'green' | 'amber' | 'red' | 'purple';
  /** Bold 16px heading in the tone's dark color */
  heading?: string;
  /** 13px line under the heading */
  subheading?: string;
  /** Dashed border for "not triggered" / inactive panels */
  dashed?: boolean;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}
export declare function TintPanel(props: TintPanelProps): JSX.Element;
