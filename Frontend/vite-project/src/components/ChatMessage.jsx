import React from "react";
import { HStack, Avatar, Box, Text } from "@chakra-ui/react";
import { FaUser, FaRobot } from "react-icons/fa";

export default function ChatMessage({ sender, text }) {
  const isUser = sender === "user";
  return (
    <HStack justify={isUser ? "flex-end" : "flex-start"}>
      {!isUser && <Avatar icon={<FaRobot />} size="sm" bg="purple.200" />}
      <Box
        bg={isUser ? "purple.500" : "gray.200"}
        color={isUser ? "white" : "black"}
        px={5}
        py={3}
        rounded="2xl"
        fontSize="md"
        boxShadow="base"
        maxW={["85%", "60%"]}
        fontFamily="Inter, Segoe UI, Arial"
      >
        {text}
      </Box>
      {isUser && <Avatar icon={<FaUser />} size="sm" bg="blue.200" />}
    </HStack>
  );
}
