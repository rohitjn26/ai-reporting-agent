const LIBRARY_URL = () => process.env.LIBRARY_URL || 'http://library:3001/v1/CUBE_CONFIG';

module.exports = {
  getLibrary: LIBRARY_URL,
  // Views live in the same library API under a different resource type. Derive the
  // VIEW endpoint from LIBRARY_URL so one env var configures both.
  getViews: () => LIBRARY_URL().replace(/\/[^/]+$/, '/VIEW'),
};
