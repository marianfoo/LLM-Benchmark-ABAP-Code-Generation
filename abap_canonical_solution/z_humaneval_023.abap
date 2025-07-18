CLASS z_humaneval_023 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS strlen
      IMPORTING
        VALUE(string) TYPE string
      RETURNING
        VALUE(length) TYPE i.

  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_023 IMPLEMENTATION.

  METHOD strlen.
    length = numofchar( string ).
  ENDMETHOD.

ENDCLASS.