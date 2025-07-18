CLASS lhc_z_humaneval_020 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    " Local type definitions
    TYPES: BEGIN OF ty_number,
             value TYPE f,
           END OF ty_number.

    TYPES: tt_numbers TYPE STANDARD TABLE OF ty_number WITH EMPTY KEY.
    TYPES: tt_result  TYPE SORTED TABLE OF f WITH UNIQUE KEY table_line.

    METHODS:
      test_closest_pair_case_1 FOR TESTING RAISING cx_static_check,
      test_closest_pair_case_2 FOR TESTING RAISING cx_static_check,
      test_closest_pair_case_3 FOR TESTING RAISING cx_static_check.

  PRIVATE SECTION.
ENDCLASS.

CLASS lhc_z_humaneval_020 IMPLEMENTATION.

  METHOD test_closest_pair_case_1.
    DATA: lt_numbers TYPE tt_numbers,
          wa_number  TYPE ty_number,
          result     TYPE tt_result,
          expected   TYPE tt_result.

    CLEAR: lt_numbers, result, expected.

    wa_number-value = CONV f( '1.0' ). APPEND wa_number TO lt_numbers.
    wa_number-value = CONV f( '2.0' ). APPEND wa_number TO lt_numbers.
    wa_number-value = CONV f( '3.9' ). APPEND wa_number TO lt_numbers.

    result = z_humaneval_020=>find_closest_elements( numbers = lt_numbers ).

    APPEND CONV f( '1.0' ) TO expected.
    APPEND CONV f( '2.0' ) TO expected.

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
      msg = 'Test case 1 failed: Expected (1.0, 2.0)' ).
  ENDMETHOD.

  METHOD test_closest_pair_case_2.
    DATA: lt_numbers TYPE tt_numbers,
          wa_number  TYPE ty_number,
          result     TYPE tt_result,
          expected   TYPE tt_result.

    CLEAR: lt_numbers, result, expected.

    wa_number-value = CONV f( '1.0' ). APPEND wa_number TO lt_numbers.
    wa_number-value = CONV f( '5.0' ). APPEND wa_number TO lt_numbers.
    wa_number-value = CONV f( '5.9' ). APPEND wa_number TO lt_numbers.

    result = z_humaneval_020=>find_closest_elements( numbers = lt_numbers ).

    APPEND CONV f( '5.0' ) TO expected.
    APPEND CONV f( '5.9' ) TO expected.

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
      msg = 'Test case 2 failed: Expected (5.0, 5.9)' ).
  ENDMETHOD.

  METHOD test_closest_pair_case_3.
    DATA: lt_numbers TYPE tt_numbers,
          wa_number  TYPE ty_number,
          result     TYPE tt_result,
          expected   TYPE tt_result.

    CLEAR: lt_numbers, result, expected.

    wa_number-value = CONV f( '2.0' ). APPEND wa_number TO lt_numbers.
    wa_number-value = CONV f( '2.2' ). APPEND wa_number TO lt_numbers.

    result = z_humaneval_020=>find_closest_elements( numbers = lt_numbers ).

    APPEND CONV f( '2.0' ) TO expected.
    APPEND CONV f( '2.2' ) TO expected.

    cl_abap_unit_assert=>assert_equals(
      act = result
      exp = expected
      msg = 'Test case 3 failed: Expected (2.0, 2.2)' ).
  ENDMETHOD.

ENDCLASS.