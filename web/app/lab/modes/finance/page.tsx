import { ModeDashboard } from '../../../../../components/ModeDashboard';
import { getDemo } from '../../../../../lib/api';

export default async function FinancePage() {
  const payload = await getDemo('finance');
  return <ModeDashboard payload={payload} />;
}
