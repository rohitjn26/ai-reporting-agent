'use strict';

// Format validation aligned with the Cube version we run (cubejs/cube:v1.6.40).
//
// 1.6.40 accepts:
//   - bare named formats: number, percent, currency, id
//   - _N-suffixed / SI named formats (NAMED_NUMERIC_FORMATS): number_0..6, percent_0..6,
//     currency_0..6, decimal(_0..6), abbr(_0..6), accounting(_0..6)
//   - any valid d3-format specifier (e.g. ",.2f", ".1%", "$,.0f", ".3s")
//   - strftime / d3-time-format strings for time dimensions
//   - imageUrl, link, or a link object for string dimensions
//
// Keep this validator in lockstep with the deployed Cube version: the named-format
// branch below requires >= the version that ships NAMED_NUMERIC_FORMATS (1.6.29+).

// Named numeric formats: bare number/percent/currency/id plus the NAMED_NUMERIC_FORMATS
// map. Suffix is a single digit 0–6 (matching the map's _0.._6 keys).
const NAMED_NUMERIC_FORMAT_REGEX = /^(number|percent|currency|decimal|abbr|accounting)(_[0-6])?$/;

function isNamedNumericFormat(value) {
  return value === 'id' || NAMED_NUMERIC_FORMAT_REGEX.test(value);
}

const STRING_DIMENSION_FORMATS = new Set(['imageUrl', 'link']);

// --- d3-format specifier validation (mirrors Cube CubeValidator) ---
// Spec: [[fill]align][sign][symbol][0][width][,][.precision][~][type]
const NUMERIC_FORMAT_TYPES = new Set(['e', 'f', 'g', 'r', 's', '%', 'p', 'b', 'o', 'd', 'x', 'X', 'c', 'n']);
const NUMERIC_FORMAT_REGEX = /^(?:(.)?([<>=^]))?([+\-( ])?([$#])?(0)?(\d+)?(,)?(?:\.(\d+))?(~)?([a-zA-Z%])?$/;

function isValidD3NumericFormat(value) {
  const match = value.match(NUMERIC_FORMAT_REGEX);
  if (!match) {
    return false;
  }
  const [, fill, align, sign, symbol, zero, width, comma, precision, tilde, type] = match;
  // A fill character requires an alignment specifier.
  if (fill && !align) {
    return false;
  }
  // Unknown type character.
  if (type && !NUMERIC_FORMAT_TYPES.has(type.toLowerCase())) {
    return false;
  }
  // Must contain at least one meaningful token.
  if (!sign && !symbol && !zero && !width && !comma && precision === undefined && !tilde && !type) {
    return false;
  }
  return true;
}

// --- strftime / d3-time-format validation (mirrors Cube CubeValidator) ---
// POSIX standard specifiers + d3-time-format extensions.
const TIME_SPECIFIERS = new Set([
  'a', 'A', 'b', 'B', 'c', 'd', 'H', 'I', 'j', 'm',
  'M', 'n', 'p', 'S', 't', 'U', 'w', 'W', 'x', 'X',
  'y', 'Y', 'Z', '%',
  'e', 'f', 'g', 'G', 'L', 'q', 'Q', 's', 'u', 'V',
]);

function isValidStrftimeFormat(value) {
  let hasSpecifier = false;
  let i = 0;
  while (i < value.length) {
    if (value[i] === '%') {
      if (i + 1 >= value.length) {
        return false; // incomplete specifier at end of string
      }
      const specifier = value[i + 1];
      if (!TIME_SPECIFIERS.has(specifier)) {
        return false; // unknown specifier
      }
      if (specifier !== '%') {
        hasSpecifier = true; // %% is a literal escape, not a date/time specifier
      }
      i += 2;
    } else {
      i++;
    }
  }
  return hasSpecifier;
}

const MEASURE_FORMAT_ERROR =
  'must be a Cube named numeric format (number, percent, currency, id, decimal, abbr, accounting — optionally with a _0–_6 suffix) or a valid d3-format specifier (e.g. ",.2f", ".1%", "$,.0f", ".3s")';

const DIMENSION_FORMAT_ERROR =
  'must be a valid Cube dimension format (imageUrl, link, a link object, a named numeric format, a d3-format specifier, or a strftime time format)';

function isValidMeasureFormat(format) {
  if (typeof format !== 'string' || !format.trim()) {
    return false;
  }
  const value = format.trim();
  if (isNamedNumericFormat(value)) {
    return true;
  }
  return isValidD3NumericFormat(value);
}

function isValidDimensionFormat(format, dimensionType) {
  if (format === null || format === undefined) {
    return true;
  }
  if (typeof format === 'object') {
    return format.type === 'link' && typeof format.label === 'string';
  }
  if (typeof format !== 'string' || !format.trim()) {
    return false;
  }
  const value = format.trim();
  if (dimensionType === 'string') {
    return STRING_DIMENSION_FORMATS.has(value);
  }
  if (dimensionType === 'time') {
    return isValidStrftimeFormat(value);
  }
  // number (and any other / default numeric-style dimension)
  if (isNamedNumericFormat(value)) {
    return true;
  }
  return isValidD3NumericFormat(value);
}

module.exports = {
  isValidMeasureFormat,
  isValidDimensionFormat,
  MEASURE_FORMAT_ERROR,
  DIMENSION_FORMAT_ERROR,
};
