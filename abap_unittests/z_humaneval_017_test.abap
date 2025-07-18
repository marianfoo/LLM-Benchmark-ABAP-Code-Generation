CLASS lhc_test DEFINITION FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS
  FINAL
  CREATE PUBLIC.

  PUBLIC SECTION.
    METHODS:
      test_parse_empty_string FOR TESTING,
      test_parse_simple_notes FOR TESTING,
      test_parse_dots FOR TESTING,
      test_parse_mixed_notes_1 FOR TESTING,
      test_parse_mixed_notes_2 FOR TESTING.

  PRIVATE SECTION.
ENDCLASS.

CLASS lhc_test IMPLEMENTATION.

  METHOD test_parse_empty_string.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_017=>parse_music( '' )
      exp = VALUE int4_table( ) ).
  ENDMETHOD.

  METHOD test_parse_simple_notes.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_017=>parse_music( 'o o o o' )
      exp = VALUE int4_table( ( 4 ) ( 4 ) ( 4 ) ( 4 ) ) ).
  ENDMETHOD.

  METHOD test_parse_dots.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_017=>parse_music( '.| .| .| .|' )
      exp = VALUE int4_table( ( 1 ) ( 1 ) ( 1 ) ( 1 ) ) ).
  ENDMETHOD.

  METHOD test_parse_mixed_notes_1.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_017=>parse_music( 'o| o| .| .| o o o o' )
      exp = VALUE int4_table( ( 2 ) ( 2 ) ( 1 ) ( 1 ) ( 4 ) ( 4 ) ( 4 ) ( 4 ) ) ).
  ENDMETHOD.

  METHOD test_parse_mixed_notes_2.
    cl_aunit_assert=>assert_equals(
      act = z_humaneval_017=>parse_music( 'o| .| o| .| o o| o o|' )
      exp = VALUE int4_table( ( 2 ) ( 1 ) ( 2 ) ( 1 ) ( 4 ) ( 2 ) ( 4 ) ( 2 ) ) ).
  ENDMETHOD.

ENDCLASS.