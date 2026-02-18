import { ModeDashboard } from '../../../../../components/ModeDashboard';
import { getDemo } from '../../../../../lib/api';

export default async function EventSandboxPage() {
  const payload = await getDemo('event-sandbox');
  return <ModeDashboard payload={payload} />;
}
