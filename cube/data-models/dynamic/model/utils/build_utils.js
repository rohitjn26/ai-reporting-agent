import {
  isValidMeasureFormat,
  isValidDimensionFormat,
  MEASURE_FORMAT_ERROR,
  DIMENSION_FORMAT_ERROR,
} from '../format-utils';

const isDescriptionEmpty = (d) =>
  d === '' || (typeof d === 'string' && d.trim() === '');

export const convertStringPropToFunction = (propNames, definition, options = {}) => {
  const result = { ...definition };
  propNames.forEach((prop) => {
    const val = result[prop];
    if (!val) return;
    result[prop] = () => val;
  });
  if (options.stripEmptyDescription && isDescriptionEmpty(result.description)) {
    delete result.description;
  }
  return result;
};

// Convert granularities from array [{name, ...}] → object {name: {...}} if needed,
// since Cube.js v2 requires the object form but the library may store either.
const normaliseGranularities = (dim) => {
  if (!dim.granularities || !Array.isArray(dim.granularities)) return dim;
  const obj = {};
  dim.granularities.forEach(({ name, ...rest }) => { if (name) obj[name] = rest; });
  return { ...dim, granularities: obj };
};

export const transformDimensions = (dimensions) =>
  Object.keys(dimensions).reduce((acc, name) => {
    acc[name] = convertStringPropToFunction(['sql'], normaliseGranularities(dimensions[name]), {
      stripEmptyDescription: true,
    });
    return acc;
  }, {});

export const transformMeasures = (measures) =>
  Object.keys(measures).reduce((acc, name) => {
    acc[name] = convertStringPropToFunction(['sql', 'drill_members'], measures[name], {
      stripEmptyDescription: true,
    });
    return acc;
  }, {});

export const transformJoins = (joins) =>
  Object.keys(joins).reduce((acc, name) => {
    acc[name] = convertStringPropToFunction(['sql'], joins[name]);
    return acc;
  }, {});

export const validateCubeSchema = (cubeName, cubeData) => {
  const errors = [];

  if (cubeData.measures) {
    Object.entries(cubeData.measures).forEach(([measureName, cfg]) => {
      if (cfg.format && !isValidMeasureFormat(cfg.format))
        errors.push(`measures.${measureName}.format ${MEASURE_FORMAT_ERROR}`);
      if (cfg.public !== undefined && typeof cfg.public !== 'boolean')
        errors.push(`measures.${measureName}.public must be a boolean`);
    });
  }

  if (cubeData.dimensions) {
    Object.entries(cubeData.dimensions).forEach(([dimName, cfg]) => {
      if (!cfg.sql && !cfg.case && !cfg.latitude && !cfg.longitude)
        errors.push(`dimensions.${dimName} does not match any allowed type`);
      if (cfg.format !== undefined && !isValidDimensionFormat(cfg.format, cfg.type))
        errors.push(`dimensions.${dimName}.format ${DIMENSION_FORMAT_ERROR}`);
    });
  }

  return errors;
};
