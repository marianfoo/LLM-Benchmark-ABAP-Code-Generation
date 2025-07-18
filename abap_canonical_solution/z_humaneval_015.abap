CLASS z_humaneval_015 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    CLASS-METHODS string_sequence
      IMPORTING
        VALUE(n)      TYPE i
      RETURNING
        VALUE(result) TYPE string.
ENDCLASS.

CLASS z_humaneval_015 IMPLEMENTATION.

  METHOD string_sequence.
    DATA: lv_result    TYPE string,
          lv_index     TYPE i,
          lv_index_str TYPE string.

    DO n + 1 TIMES.
      lv_index = sy-index - 1.
      lv_index_str = lv_index.
      CONCATENATE lv_result lv_index_str INTO lv_result SEPARATED BY space.
    ENDDO.

    CONDENSE lv_result.
    result = lv_result.
  ENDMETHOD.

ENDCLASS.