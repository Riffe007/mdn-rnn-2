import Link from 'next/link';

import type { ModeCard as ModeCardType } from '../lib/modes';

export function ModeCard({ mode }: { mode: ModeCardType }) {
  return (
    <Link className="panel mode-card" href={`/lab/modes/${mode.slug}`}>
      <h3>{mode.title}</h3>
      <p>{mode.description}</p>
    </Link>
  );
}
