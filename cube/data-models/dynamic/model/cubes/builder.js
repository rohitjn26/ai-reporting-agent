import {
  transformDimensions,
  transformMeasures,
  transformJoins,
  transformView,
  validateCubeSchema,
  validateViewSchema,
} from './model/utils/build_utils.js';

const { getLibrary, getViews } = require('../library');
const { log, error } = require('../debug');
const fetch = require('node-fetch');

const fetchResources = async (url) => {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Library responded ${resp.status} for ${url}`);
  }
  const { data = [] } = await resp.json();
  return data;
};

asyncModule(async () => {
  // 1) Base cubes — single-table, with joins declared here so views can traverse
  //    them. These are private (public:false); the agent never sees them directly.
  try {
    log('Fetching cube configs from', getLibrary());
    const resources = await fetchResources(getLibrary());
    log(`Received ${resources.length} cube config(s)`);

    for (const resource of resources) {
      const cubeData = resource.data;
      const cubeName = (cubeData && cubeData.name) || resource.name;

      if (!cubeName || !cubeData) {
        log('Skipping resource — missing name or data:', resource.id);
        continue;
      }

      const validationErrors = validateCubeSchema(cubeName, cubeData);
      if (validationErrors.length > 0) {
        log(`Skipping ${cubeName} — validation errors:`, validationErrors.join(', '));
        continue;
      }

      try {
        cube(cubeName, {
          sql: () => cubeData.sql,
          public: cubeData.public !== false,
          refreshKey: { every: '10 second' },
          dimensions: transformDimensions(cubeData.dimensions || {}),
          measures:   transformMeasures(cubeData.measures   || {}),
          joins:      transformJoins(cubeData.joins         || {}),
        });
        log('Registered cube:', cubeName);
      } catch (err) {
        error(`Failed to register cube ${cubeName}:`, err.message);
      }
    }
  } catch (err) {
    error('Failed to load cube configs:', err.message);
    throw err;
  }

  // 2) Views — the only public surface. Joins are resolved through the cubes'
  //    join_path declarations above, so no join SQL is exposed. A missing VIEW
  //    endpoint or zero views must not break cube loading, so this stays non-fatal.
  try {
    log('Fetching view configs from', getViews());
    const views = await fetchResources(getViews());
    log(`Received ${views.length} view config(s)`);

    for (const resource of views) {
      const viewData = resource.data;
      const viewName = (viewData && viewData.name) || resource.name;

      if (!viewName || !viewData) {
        log('Skipping view — missing name or data:', resource.id);
        continue;
      }

      const validationErrors = validateViewSchema(viewName, viewData);
      if (validationErrors.length > 0) {
        log(`Skipping view ${viewName} — validation errors:`, validationErrors.join(', '));
        continue;
      }

      try {
        view(viewName, transformView(viewData));
        log('Registered view:', viewName);
      } catch (err) {
        error(`Failed to register view ${viewName}:`, err.message);
      }
    }
  } catch (err) {
    error('Failed to load view configs (continuing without views):', err.message);
  }
});
