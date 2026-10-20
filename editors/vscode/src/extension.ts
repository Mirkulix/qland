import * as path from 'path';
import * as fs from 'fs';
import { workspace, ExtensionContext, window } from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions
} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: ExtensionContext) {
    const config = workspace.getConfiguration('qal');
    let binaryPath = config.get<string>('path') || 'qal';

    // Auto-detect bin/qal in workspace if default 'qal' is not found
    if (binaryPath === 'qal') {
        const workspaceFolders = workspace.workspaceFolders;
        if (workspaceFolders && workspaceFolders.length > 0) {
            const root = workspaceFolders[0].uri.fsPath;
            const localBin = path.join(root, 'bin', 'qal');
            if (fs.existsSync(localBin)) {
                binaryPath = localBin;
            }
        }
    }

    const serverOptions: ServerOptions = {
        command: binaryPath,
        args: ['lsp'],
        options: {
            env: process.env
        }
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'qal' }],
        synchronize: {
            fileEvents: workspace.createFileSystemWatcher('**/*.qal')
        }
    };

    client = new LanguageClient(
        'qalLsp',
        'QAL Language Server',
        serverOptions,
        clientOptions
    );

    client.start().catch(err => {
        window.showErrorMessage(`Failed to start QAL Language Server (${binaryPath}): ${err}`);
    });
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
