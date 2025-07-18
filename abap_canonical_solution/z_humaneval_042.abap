CLASS z_humaneval_042 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: incr_list
      IMPORTING
        VALUE(l)      TYPE int4_table
      RETURNING
        VALUE(result) TYPE int4_table.

  PROTECTED SECTION.
  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_042 IMPLEMENTATION.
  METHOD incr_list.
    LOOP AT l INTO DATA(e).
      result = VALUE #( BASE result ( e + 1 ) ).
    ENDLOOP.
  ENDMETHOD.
ENDCLASS.