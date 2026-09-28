// User Login Workflow & Route Handler
import { authenticateUser } from './auth';
import { db } from './db';

export interface LoginRequest {
  email: string;
  passwordHash: string;
}

export interface LoginResponse {
  statusCode: number;
  message: string;
  sessionToken?: string;
}

/**
 * Handles the end-to-end user login process.
 * 1. Validates incoming request credentials
 * 2. Invokes authenticateUser for credential verification
 * 3. Records login audit trail in database
 * 4. Returns session token to client
 */
export async function handleLogin(request: LoginRequest): Promise<LoginResponse> {
  if (!request.email || !request.passwordHash) {
    return { statusCode: 400, message: 'Missing credentials' };
  }

  // Authenticate user against auth service
  const authResult = await authenticateUser(request.email, request.passwordHash);

  if (!authResult.success) {
    return { statusCode: 401, message: authResult.error || 'Authentication failed' };
  }

  // Update last login timestamp in database
  await db.query('UPDATE users SET last_login = ? WHERE id = ?', [
    Date.now(),
    authResult.userId,
  ]);

  return {
    statusCode: 200,
    message: 'Login successful',
    sessionToken: authResult.token,
  };
}
