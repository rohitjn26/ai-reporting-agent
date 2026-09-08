const { FileRepository } = require('@cubejs-backend/server-core');
const PostgresDriver = require('@cubejs-backend/postgres-driver');

module.exports = {
  driverFactory: () =>
    new PostgresDriver({
      host:     process.env.CUBEJS_DB_HOST     || 'postgres',
      port:     parseInt(process.env.CUBEJS_DB_PORT || '5432'),
      database: process.env.CUBEJS_DB_NAME     || 'reporting',
      user:     process.env.CUBEJS_DB_USER     || 'postgres',
      password: process.env.CUBEJS_DB_PASS     || 'postgres',
    }),

  checkAuth: () => {},

  queryRewrite: (query) => query,

  repositoryFactory: () =>
    new FileRepository('data-models/dynamic'),

  http: {
    cors: {
      origin: '*',
      credentials: true,
      methods: 'GET,HEAD,PUT,PATCH,POST,DELETE',
      allowedHeaders: ['Content-Type', 'Authorization', 'x-request-id'],
      optionsSuccessStatus: 204,
    },
  },

  logger: (msg, params) => {
    const safe = { ...params };
    delete safe.securityContext;
    console.log(`[cube] ${msg}:`, JSON.stringify(safe));
  },
};
