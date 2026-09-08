module.exports = {
  getLibrary: () => process.env.LIBRARY_URL || 'http://library:3001/v1/CUBE_CONFIG',
};
