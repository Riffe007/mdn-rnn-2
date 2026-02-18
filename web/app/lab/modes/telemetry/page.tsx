import { ModeDashboard } from '../../../../../components/ModeDashboard';
import { getDemo } from '../../../../../lib/api';

export default async function TelemetryPage() {
  const payload = await getDemo('telemetry');
  return <ModeDashboard payload={payload} />;
}
