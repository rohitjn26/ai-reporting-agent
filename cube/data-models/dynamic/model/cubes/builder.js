import { transformDimensions, transformMeasures, transformJoins, validateCubeSchema } from './model/utils/build_utils.js';

const { getLibrary } = require('../library');
const { log, error } = require('../debug');
const fetch = require('node-fetch');

const libraryUrl = getLibrary();

asyncModule(async () => {
  try {
    log('Fetching cube configs from', libraryUrl);

    const resp = await fetch(libraryUrl);
    if (!resp.ok) {
      throw new Error(`Library responded ${resp.status}`);
    }

    const { data: resources = [] } = await resp.json();
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
});
