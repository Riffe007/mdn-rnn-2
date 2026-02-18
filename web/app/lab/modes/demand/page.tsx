import { ModeDashboard } from '../../../../../components/ModeDashboard';
import { getDemo } from '../../../../../lib/api';

export default async function DemandPage() {
  const payload = await getDemo('demand');
  return <ModeDashboard payload={payload} />;
}
