// Sample TypeScript File
interface User {
  id: string;
  name: string;
  email: string;
}

export class UserService {
  private users: Map<string, User> = new Map();

  public register(user: User): boolean {
    if (this.users.has(user.id)) {
      return false;
    }
    this.users.set(user.id, user);
    return true;
  }

  public findById(id: string): User | undefined {
    return this.users.get(id);
  }
}
