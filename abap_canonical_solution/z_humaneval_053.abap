CLASS z_humaneval_053 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: add
      IMPORTING
        x             TYPE i
        y             TYPE i
      RETURNING
        VALUE(result) TYPE i.
  PROTECTED SECTION.
  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_053 IMPLEMENTATION.
  METHOD add.
    result = x + y.
  ENDMETHOD.
ENDCLASS.