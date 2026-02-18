import Link from 'next/link';

import { getModeCards } from '../../lib/api';

export default async function LabHomePage() {
  const cards = await getModeCards();

  return (
    <main>
      <section className="panel grid">
        <h1>PRE Multi-Mode Lab</h1>
        <p>
          One inference engine, multiple wrappers. Each mode focuses on distinct risk narratives
          while preserving a consistent probabilistic contract.
        </p>
      </section>
      <section className="mode-grid" style={{ marginTop: '1rem' }}>
        {cards.map((mode) => (
          <Link className="panel mode-card" key={mode.mode} href={`/lab/modes/${mode.mode}`}>
            <h3>{mode.title}</h3>
            <p>{mode.description}</p>
            <p className="kpi">{mode.dataset} · {mode.model}</p>
          </Link>
        ))}
      </section>
    </main>
  );
}
