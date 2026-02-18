import { ModeDashboard } from '../../../../../components/ModeDashboard';
import { getDemo } from '../../../../../lib/api';

export default async function ProjectRiskPage() {
  const payload = await getDemo('project-risk');
  return <ModeDashboard payload={payload} />;
}
