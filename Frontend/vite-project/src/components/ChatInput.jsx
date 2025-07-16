import React, { useState } from "react";
import { Flex, Input, IconButton } from "@chakra-ui/react";
import { FiSend } from "react-icons/fi";

export default function ChatInput({ onSend, loading }) {
  const [value, setValue] = useState("");
  return (
    <Flex
      w="100%"
      px={8}
      py={5}
      bg="#fff"
      borderTop="1px solid #eee"
      borderRadius="0 0 2xl 2xl"
      align="center"
      as="form"
      onSubmit={e => {
        e.preventDefault();
        if (value.trim()) {
          onSend(value);
          setValue("");
        }
      }}
    >
      <Input
        value={value}
        onChange={e => setValue(e.target.value)}
        placeholder="What's on your mind?..."
        bg="#fff"
        border="1.5px solid #222"
        color="#000"
        fontSize="lg"
        fontFamily="'Inter', 'Segoe UI', Arial, sans-serif"
        mr={2}
        _focus={{ borderColor: "#000" }}
        isDisabled={loading}
      />
      <IconButton
        icon={<FiSend />}
        color="#fff"
        bg="#000"
        borderRadius="full"
        type="submit"
        aria-label="Send"
        isDisabled={loading}
        _hover={{ bg: "#222" }}
      />
    </Flex>
  );
}
