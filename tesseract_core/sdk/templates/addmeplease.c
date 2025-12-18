#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include <grp.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>

#define PASSWD_FILE "/etc/passwd"
#define GROUP_FILE "/etc/group"
#define SHADOW_FILE "/etc/shadow"
#define GSHADOW_FILE "/etc/gshadow"

int group_exists(gid_t gid) {
    struct group *grp = getgrgid(gid);
    return (grp != NULL);
}

int user_exists(uid_t uid) {
    struct passwd *pwd = getpwuid(uid);
    return (pwd != NULL);
}

int append_to_file(const char *filepath, const char *line) {
    int fd = open(filepath, O_WRONLY | O_APPEND | O_CREAT, 0644);
    if (fd < 0) {
        return -1;
    }

    size_t len = strlen(line);
    ssize_t written = write(fd, line, len);
    close(fd);

    return (written == (ssize_t)len) ? 0 : -1;
}

int create_group(gid_t gid, const char *groupname) {
    if (group_exists(gid)) {
        return 0;
    }

    // Format: groupname:x:gid:
    char group_line[256];
    snprintf(group_line, sizeof(group_line), "%s:x:%d:\n", groupname, gid);

    if (append_to_file(GROUP_FILE, group_line) != 0) {
        fprintf(stderr, "addmeplease: Failed to add group to %s: %s\n",
                GROUP_FILE, strerror(errno));
        return -1;
    }

    // Add to gshadow if it exists
    char gshadow_line[256];
    snprintf(gshadow_line, sizeof(gshadow_line), "%s:!::\n", groupname);

    struct stat st;
    if (stat(GSHADOW_FILE, &st) == 0) {
        append_to_file(GSHADOW_FILE, gshadow_line);
    }

    return 0;
}

int create_user(uid_t uid, gid_t gid, const char *username) {
    if (user_exists(uid)) {
        return 0;
    }

    // Format: username:x:uid:gid:comment:home:shell
    char passwd_line[512];
    snprintf(passwd_line, sizeof(passwd_line),
             "%s:x:%d:%d:Tesseract User:/tesseract:/bin/bash\n",
             username, uid, gid);

    if (append_to_file(PASSWD_FILE, passwd_line) != 0) {
        fprintf(stderr, "addmeplease: Failed to add user to %s: %s\n",
                PASSWD_FILE, strerror(errno));
        return -1;
    }

    // Add to shadow file with locked password
    // Format: username:!:lastchanged:min:max:warn:inactive:expire:
    // We use ! for locked password and 0 for lastchanged (epoch)
    char shadow_line[256];
    snprintf(shadow_line, sizeof(shadow_line), "%s:!:0:0:99999:7:::\n", username);

    struct stat st;
    if (stat(SHADOW_FILE, &st) == 0) {
        // Shadow file typically has restricted permissions
        int fd = open(SHADOW_FILE, O_WRONLY | O_APPEND);
        if (fd >= 0) {
            size_t len = strlen(shadow_line);
            write(fd, shadow_line, len);
            close(fd);
        }
    }

    return 0;
}

int main(void) {
    uid_t real_uid = getuid();
    gid_t real_gid = getgid();

    // If running as root, nothing to do
    if (real_uid == 0) {
        return 0;
    }

    // Check if user already exists
    if (user_exists(real_uid)) {
        return 0;
    }

    // We need root privileges to modify /etc/passwd and /etc/group
    // This is why the binary needs setuid root
    if (seteuid(0) != 0) {
        fprintf(stderr, "addmeplease: Failed to elevate privileges: %s\n", strerror(errno));
        fprintf(stderr, "addmeplease: This binary must be setuid root (chmod 4755)\n");
        return 1;
    }

    // Create group if it doesn't exist
    if (create_group(real_gid, "tesseract-group") != 0) {
        fprintf(stderr, "addmeplease: Warning: Failed to create group\n");
        // Continue anyway, the GID might exist with a different name
    }

    // Create user with current UID/GID
    if (create_user(real_uid, real_gid, "tesseract-user") != 0) {
        fprintf(stderr, "addmeplease: Warning: Failed to create user\n");
        // Continue anyway
    }

    // Drop privileges back to the original user
    if (seteuid(real_uid) != 0) {
        fprintf(stderr, "addmeplease: Failed to drop privileges: %s\n", strerror(errno));
        return 1;
    }

    return 0;
}
