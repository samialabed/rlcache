from abc import ABC


class RLConverter(ABC):
    """
    Abstract MDP converter.

    Converts between system and agent view of states, actions, and rewards, generally
    using a `Schema`.
    """

    def system_to_agent_state(self, *args, **kwargs):
        """
        Converts system state to agent state.
        """
        raise NotImplementedError

    def system_to_agent_action(self, *args, **kwargs):
        """
        Converts system output to an agent action.
        """
        raise NotImplementedError

    def agent_to_system_action(self, actions, **kwargs):
        """
        This maps agent output to a system action.

        Args:
            actions: Action dict.

        Returns:
            any: System command to execute action.
        """
        raise NotImplementedError

    def system_to_agent_reward(self, *args, **kwargs):
        """
        Converts system observations to scalar reward.

        Returns:
            float: Reward.
        """

        raise NotImplementedError
