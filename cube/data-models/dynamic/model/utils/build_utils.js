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

// Join SQL contains Cube references like `${CUBE}.customer_id = ${customers.id}`.
// Cube's transpiler normally turns such a template literal into a function whose
// params are the referenced cubes (CUBE, customers) so `${...}` interpolates the
// proxy args into real SQL. We build joins dynamically from JSON strings, so
// reproduce that: collect the base identifiers referenced via ${...} and return a
// function that interpolates them through a template literal. (Plain `() => sql`
// would leave the ${...} literal — which is why joins previously errored.)
const sqlRefToFunction = (sqlString) => {
  const refs = [];
  const re = /\$\{\s*([a-zA-Z_][a-zA-Z0-9_]*)/g;
  let m;
  while ((m = re.exec(sqlString)) !== null) {
    if (!refs.includes(m[1])) refs.push(m[1]);
  }
  // eslint-disable-next-line no-new-func
  return new Function(...refs, 'return `' + sqlString.replace(/`/g, '\\`') + '`;');
};

export const transformJoins = (joins) =>
  Object.keys(joins).reduce((acc, name) => {
    const def = { ...joins[name] };
    if (typeof def.sql === 'string') def.sql = sqlRefToFunction(def.sql);
    acc[name] = def;
    return acc;
  }, {});

// In JS models Cube's transpiler rewrites `join_path: orders.customers` into an
// arrow function `(orders) => orders.customers` (param = base cube, body = the
// join traversal). We build views dynamically and so bypass that transpile step,
// so reproduce the function form here from the "a.b.c" string. `includes`/`excludes`
// arrays are accepted as-is by the view validator, so they pass through untouched.
const joinPathToFunction = (joinPath) => {
  if (typeof joinPath !== 'string') return joinPath; // already a function
  const base = joinPath.split('.')[0];
  // eslint-disable-next-line no-new-func
  return new Function(base, `return ${joinPath};`);
};

export const transformView = (viewData) => {
  const cubes = (viewData.cubes || []).map((entry) => ({
    ...entry,
    join_path: joinPathToFunction(entry.join_path),
  }));
  const view = { cubes };
  if (!isDescriptionEmpty(viewData.description)) view.description = viewData.description;
  if (viewData.title) view.title = viewData.title;
  if (viewData.public !== undefined) view.public = viewData.public;
  return view;
};

export const validateViewSchema = (viewName, viewData) => {
  const errors = [];
  if (!Array.isArray(viewData.cubes) || viewData.cubes.length === 0) {
    errors.push('view must have a non-empty `cubes` array');
    return errors;
  }
  viewData.cubes.forEach((entry, i) => {
    if (!entry.join_path || typeof entry.join_path !== 'string')
      errors.push(`cubes[${i}] needs a string join_path`);
    const inc = entry.includes;
    const hasIncludes = inc === '*' || (Array.isArray(inc) && inc.length > 0);
    if (!hasIncludes)
      errors.push(`cubes[${i}] needs includes (a member list or "*")`);
  });
  return errors;
};

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
