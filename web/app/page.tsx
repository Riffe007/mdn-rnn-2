import Link from 'next/link';

export default function HomePage() {
  return (
    <main>
      <section className="panel grid">
        <h1>Riffe Labs Probabilistic Risk Engine</h1>
        <p>
          Distribution-first forecasting and uncertainty modeling across operational,
          demand, project, event, and financial regime demos.
        </p>
        <Link className="panel" href="/lab">
          Enter Lab
        </Link>
      </section>
    </main>
  );
}
