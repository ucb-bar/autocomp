#include <stdio.h>
#include <zephyr/sys/reboot.h>

int main(void)
{
        // Print to the host via TSI
        printf("Hello World! %s\n", CONFIG_BOARD_TARGET);

        // Send an exit command to the host to terminate the simulation
        sys_reboot(SYS_REBOOT_COLD);
        return 0;
}