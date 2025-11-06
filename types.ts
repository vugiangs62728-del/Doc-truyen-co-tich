export enum ConnectionState {
  IDLE = 'IDLE',
  CONNECTING = 'CONNECTING',
  CONNECTED = 'CONNECTED',
  DISCONNECTED = 'DISCONNECTED',
  ERROR = 'ERROR',
}

export interface Transcript {
  id: number;
  speaker: 'user' | 'model';
  text: string;
  isFinal: boolean;
}
