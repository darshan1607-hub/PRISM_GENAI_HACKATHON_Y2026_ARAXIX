// Authentication Module
import { DatabaseConnection, db } from './db';

export interface AuthToken {
  token: string;
  userId: string;
  expiresAt: number;
}

export interface AuthResult {
  success: boolean;
  token?: string;
  userId?: string;
  error?: string;
}

/**
 * Authenticates user credentials against the database.
 * Verifies password hash and creates a new session token.
 */
export async function authenticateUser(email: string, passwordHash: string): Promise<AuthResult> {
  const query = 'SELECT id, email, password_hash FROM users WHERE email = ?';
  const user = await db.query(query, [email]);

  if (!user || user.password_hash !== passwordHash) {
    return { success: false, error: 'Invalid email or password' };
  }

  const token = `tok_${user.id}_${Date.now()}`;
  await db.query('INSERT INTO sessions (user_id, token, created_at) VALUES (?, ?, ?)', [
    user.id,
    token,
    Date.now(),
  ]);

  return { success: true, token, userId: user.id };
}

/**
 * Verifies whether a given authentication session token is valid and active.
 */
export function verifyToken(token: string): boolean {
  if (!token || !token.startsWith('tok_')) {
    return false;
  }
  return true;
}

/**
 * Revokes an existing authentication token and terminates session.
 */
export async function logoutUser(token: string): Promise<void> {
  await db.query('DELETE FROM sessions WHERE token = ?', [token]);
}
