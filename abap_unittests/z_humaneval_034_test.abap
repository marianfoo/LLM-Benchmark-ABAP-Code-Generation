CLASS lt_humaneval_034 DEFINITION
  FINAL
  FOR TESTING
    RISK LEVEL HARMLESS
  CREATE PUBLIC.
  PUBLIC SECTION.
    METHODS: check FOR TESTING.
  PROTECTED SECTION.
  PRIVATE SECTION.
ENDCLASS.

CLASS lt_humaneval_034 IMPLEMENTATION.

  METHOD check.
    DATA(result) = z_humaneval_034=>unique( it_list = VALUE int4_table(
      ( 5 ) ( 3 ) ( 5 ) ( 2 ) ( 3 ) ( 3 ) ( 9 ) ( 0 ) ( 123 ) ) ).

    DATA(expected) = VALUE int4_table( ( 0 ) ( 2 ) ( 3 ) ( 5 ) ( 9 ) ( 123 ) ).

  ENDMETHOD.

ENDCLASS.